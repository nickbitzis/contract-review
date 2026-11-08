import io

import pytest
from docx import Document
from fastapi.testclient import TestClient

from contractreview.api import create_app
from contractreview.docx_io import DocxError, docx_to_text


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(pipeline))


def make_docx() -> bytes:
    d = Document()
    d.add_heading("Master Services Agreement", 0)
    d.add_paragraph('between Brightwater Mills ("Customer") and Tessavel, Inc. ("Vendor")')
    d.add_heading("Fees and Payment", 1)
    d.add_paragraph("Customer shall pay all undisputed invoices within ninety (90) days of receipt.")
    d.add_heading("Governing Law", 1)
    d.add_paragraph("This Agreement is governed by the laws of the State of Delaware.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Insurance", "Vendor shall maintain insurance of $20 million."
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_docx_ingestion():
    text = docx_to_text(make_docx())
    assert text.startswith("Master Services Agreement\n") and "# Fees and Payment" in text
    assert "Insurance: Vendor shall maintain" in text
    with pytest.raises(DocxError):
        docx_to_text(b"not a zip")


def test_health_metrics_index_playbook(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["mode"] == "baseline+parser" and h["playbook_clause_types"] == 12
    assert "Contract Review" in client.get("/").text
    assert client.get("/api/playbook").json()["clauses"]["insurance"]["label"] == "Insurance"
    assert len(client.get("/api/samples").json()) == 40


def test_end_to_end_review(client):
    doc = client.post("/api/contracts/sample/msa-04").json()
    cid = doc["id"]
    assert doc["summary"]["issues"] == len(doc["issues"]) > 0 and doc["status"] == "in_review"
    first, second = doc["issues"][0], doc["issues"][1]
    r = client.post(f"/api/contracts/{cid}/issues/{first['id']}/reject", json={"reviewer": "ana"}).json()
    assert r["status"] == "in_review" and r["issue"]["required_approvals"]
    role = r["issue"]["required_approvals"][0]
    r = client.post(f"/api/contracts/{cid}/issues/{first['id']}/approvals", json={"role": role, "approver": "lee"})
    assert r.status_code == 200 and role in r.json()["issue"]["approvals"]
    r = client.post(f"/api/contracts/{cid}/issues/{first['id']}/approvals", json={"role": "nobody", "approver": "x"})
    assert r.status_code == 409
    r = client.post(f"/api/contracts/{cid}/issues/{second['id']}/accept", json={"reviewer": "ana"})
    assert r.json()["issue"]["status"] == "accepted"
    assert client.post(f"/api/contracts/{cid}/issues/{second['id']}/shrug", json={}).status_code == 422
    assert client.post(f"/api/contracts/{cid}/issues/I99/accept", json={}).status_code == 404
    audit = client.get(f"/api/contracts/{cid}/audit").json()
    assert [e["action"] for e in audit][:3] == ["analyzed", "reject", "routed"]
    memo = client.get(f"/api/contracts/{cid}/memo.md")
    assert memo.status_code == 200 and "review-memo.md" in memo.headers["content-disposition"]
    assert "<ins>" in client.get(f"/api/contracts/{cid}/redline.html").text
    assert client.get(f"/api/contracts/{cid}/clean.txt").status_code == 200
    assert client.get("/api/contracts").json()[0]["id"] == cid
    m = client.get("/metrics").text
    assert "contractreview_decisions_total" in m and "contractreview_issues_total" in m


def test_uploads(client):
    r = client.post("/api/contracts", files={"file": ("msa.docx", make_docx(), "application/octet-stream")})
    assert r.status_code == 200
    params = r.json()["params"]
    assert params["payment_terms"]["payment_days"]["value"] == 90
    assert params["governing_law"]["law"]["value"] == "Delaware"
    sev = {(i["type"], i["param"]): i["severity"] for i in r.json()["issues"]}
    assert sev[("payment_terms", "payment_days")] == "high"
    r = client.post("/api/contracts/text", json={"text": "ORDER FORM\n\nCustomer: X\nPayment Terms: Net 60\n"})
    assert r.json()["kind"] == "order_form" and r.json()["issues"][0]["param"] == "payment_days"
    txt = client.post("/api/contracts", files={"file": ("a.txt", b"AGREEMENT\n\n1. Payment\nPayment terms are net 30.\n",
                                                        "text/plain")})
    assert txt.status_code == 200


def test_upload_errors(client):
    assert client.post("/api/contracts", files={"file": ("a.pdf", b"%PDF", "application/pdf")}).status_code == 415
    assert client.post("/api/contracts", files={"file": ("a.docx", b"junk", "application/octet-stream")}).status_code == 422
    assert client.post("/api/contracts", files={"file": ("a.txt", b"\xff\xfe\x00", "text/plain")}).status_code == 422
    assert client.post("/api/contracts/sample/nope").status_code == 404
    assert client.get("/api/contracts/missing").status_code == 404
    assert client.post("/api/contracts/text", json={"text": "short"}).status_code == 422
