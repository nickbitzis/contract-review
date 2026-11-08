"""FastAPI service: upload and analyze contracts, review actions, exports, /healthz, /metrics, and the web UI."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from . import __version__
from .analyze import Pipeline
from .corpus import build_corpus
from .docx_io import DocxError, docx_to_text
from .logs import event, get_logger, now_ms
from .review import ReviewError, Store

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 5 * 1024 * 1024


class TextBody(BaseModel):
    text: str = Field(min_length=20, max_length=500_000)
    title: str | None = Field(default=None, max_length=200)


class DecisionBody(BaseModel):
    reviewer: str = Field(default="reviewer", min_length=1, max_length=80)
    text: str | None = Field(default=None, max_length=20_000)
    note: str = Field(default="", max_length=2000)


class ApprovalBody(BaseModel):
    role: str = Field(min_length=1, max_length=40)
    approver: str = Field(min_length=1, max_length=80)
    approve: bool = True
    note: str = Field(default="", max_length=2000)


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter("contractreview_requests_total", "HTTP requests", ["path", "status"], registry=r)
        self.contracts = Counter("contractreview_contracts_analyzed_total", "Contracts analyzed", ["kind"], registry=r)
        self.issues = Counter("contractreview_issues_total", "Issues found", ["severity"], registry=r)
        self.decisions = Counter("contractreview_decisions_total", "Reviewer decisions", ["decision"], registry=r)
        self.approvals = Counter("contractreview_approvals_total", "Approver sign-offs", ["role", "result"],
                                 registry=r)
        self.analyze_seconds = Histogram("contractreview_analyze_seconds", "Time to analyze one contract",
                                         registry=r, buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))


@lru_cache(maxsize=1)
def samples() -> dict[str, dict]:
    return {c.id: {"id": c.id, "title": c.title, "kind": c.kind, "text": c.text} for c in build_corpus()}


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    pipeline = pipeline or Pipeline()
    store = Store(pipeline)
    metrics, log = Metrics(), get_logger()
    app = FastAPI(title="contractreview", version=__version__)
    app.state.store, app.state.metrics = store, metrics
    known = {"/", "/healthz", "/metrics"}

    @app.middleware("http")
    async def count(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        path = request.url.path
        label = path if path in known else ("/api" if path.startswith("/api") else "other")
        metrics.requests.labels(label, str(response.status_code)).inc()
        if path.startswith("/api"):
            event(log, "request", method=request.method, path=path, status=response.status_code,
                  latency_ms=now_ms(start))
        return response

    def doc_or_404(cid: str) -> dict:
        try:
            return store.get(cid)
        except KeyError:
            raise HTTPException(404, "contract not found") from None

    def ingest(text: str, title: str | None, source: str) -> dict:
        start = time.perf_counter()
        doc = store.create(text, title, source)
        metrics.analyze_seconds.observe(time.perf_counter() - start)
        metrics.contracts.labels(doc["kind"]).inc()
        for it in doc["issues"]:
            metrics.issues.labels(it["severity"]).inc()
        event(log, "analyzed", contract_id=doc["id"], kind=doc["kind"], source=source, issues=len(doc["issues"]),
              risk_score=doc["summary"]["risk_score"], mode=pipeline.mode, latency_ms=now_ms(start))
        return doc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__, "mode": pipeline.mode,
                "playbook_clause_types": len(pipeline.playbook.rules), "contracts": len(store.items)}

    @app.get("/metrics")
    def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/playbook")
    def playbook() -> dict:
        return pipeline.playbook.to_dict()

    @app.get("/api/samples")
    def list_samples() -> list[dict]:
        return [{k: v for k, v in s.items() if k != "text"} for s in samples().values()]

    @app.post("/api/contracts/sample/{sample_id}")
    def load_sample(sample_id: str) -> dict:
        s = samples().get(sample_id)
        if s is None:
            raise HTTPException(404, "unknown sample")
        return ingest(s["text"], s["title"], f"sample:{sample_id}")

    @app.post("/api/contracts/text")
    def upload_text(body: TextBody) -> dict:
        return ingest(body.text, body.title, "text")

    @app.post("/api/contracts")
    async def upload(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, "file is larger than 5 MB")
        name = (file.filename or "upload.txt").lower()
        if name.endswith(".docx"):
            try:
                text = docx_to_text(content)
            except DocxError as exc:
                raise HTTPException(422, str(exc)) from None
        elif name.endswith((".txt", ".md")):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(422, "text files must be UTF-8") from None
        else:
            raise HTTPException(415, "upload a .txt, .md, or .docx file")
        if len(text.strip()) < 20:
            raise HTTPException(422, "the file has no contract text")
        return ingest(text, None, f"file:{file.filename}")

    @app.get("/api/contracts")
    def list_contracts() -> list[dict]:
        return store.list()

    @app.get("/api/contracts/{cid}")
    def get_contract(cid: str) -> dict:
        return doc_or_404(cid)

    @app.post("/api/contracts/{cid}/issues/{issue_id}/approvals")
    def approve(cid: str, issue_id: str, body: ApprovalBody) -> dict:
        doc_or_404(cid)
        try:
            it = store.approve(cid, issue_id, body.role, body.approver, body.approve, body.note)
        except KeyError:
            raise HTTPException(404, "issue not found") from None
        except ReviewError as exc:
            raise HTTPException(409, str(exc)) from None
        metrics.approvals.labels(body.role, "approved" if body.approve else "declined").inc()
        event(log, "approval", contract_id=cid, issue_id=issue_id, role=body.role, approver=body.approver,
              approve=body.approve)
        return {"issue": it, "status": store.get(cid)["status"], "progress": store.get(cid)["progress"]}

    @app.post("/api/contracts/{cid}/issues/{issue_id}/{decision}")
    def decide(cid: str, issue_id: str, decision: str, body: DecisionBody) -> dict:
        doc_or_404(cid)
        try:
            it = store.decide(cid, issue_id, decision, body.reviewer, body.text, body.note)
        except KeyError:
            raise HTTPException(404, "issue not found") from None
        except ReviewError as exc:
            raise HTTPException(422, str(exc)) from None
        metrics.decisions.labels(decision).inc()
        event(log, "decision", contract_id=cid, issue_id=issue_id, decision=decision, reviewer=body.reviewer,
              approvals=it["required_approvals"])
        return {"issue": it, "status": store.get(cid)["status"], "progress": store.get(cid)["progress"]}

    @app.get("/api/contracts/{cid}/audit")
    def audit(cid: str) -> list[dict]:
        return doc_or_404(cid)["audit"]

    @app.get("/api/contracts/{cid}/memo.md")
    def memo(cid: str) -> PlainTextResponse:
        doc_or_404(cid)
        return PlainTextResponse(store.memo(cid), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{cid}-review-memo.md"'})

    @app.get("/api/contracts/{cid}/redline.html", response_class=HTMLResponse)
    def redline(cid: str, pending: bool = False) -> str:
        doc_or_404(cid)
        return store.redline(cid, include_pending=pending)

    @app.get("/api/contracts/{cid}/clean.txt")
    def clean(cid: str) -> PlainTextResponse:
        doc_or_404(cid)
        return PlainTextResponse(store.clean_text(cid))

    return app


def app_from_env() -> FastAPI:  # pragma: no cover - used by the server command
    mode = os.environ.get("CONTRACTREVIEW_MODE", "baseline")
    if mode == "claude":
        from .classify import ClaudeClassifier
        from .extract import ClaudeExtractor

        return create_app(Pipeline(classifier=ClaudeClassifier(), extractor=ClaudeExtractor()))
    return create_app()
