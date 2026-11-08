"""The review pipeline: segment, classify, extract, compare to the playbook, score, and draft redlines."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from .classify import TYPES, BaselineClassifier, Classifier
from .extract import Evidence, Extractor
from .playbook import RANK, Playbook, load_playbook, worst
from .segment import Clause, Segmented, segment

WEIGHTS = {"low": 3, "medium": 10, "high": 25}
REFERENCE = re.compile(r"subject to the (?:limitations|terms|exclusions) (?:set forth |set out |stated )?in the "
                       r"(?:Agreement|Master|MSA)", re.I)
HEADING_HITS = {
    "limitation_of_liability": ["liability", "limitation", "recovery"],
}


def describe(param: str, value) -> str:
    """Human wording for a parameter value, used in reasons and the memo."""
    if value is None:
        return "not stated"
    if param == "cap_multiple":
        return f"{value:g}x annual fees ({int(round(value * 12))} months)"
    if param in ("tfc_notice_days", "cure_days", "payment_days", "renewal_notice_days", "warranty_days"):
        return f"{value} days" if param != "payment_days" else f"net {value}"
    if param == "late_fee_pct":
        return "no late interest" if value == 0 else f"{value:g}% per month"
    if param == "breach_hours":
        return f"{value} hours"
    if param == "term_years":
        return "perpetual" if value >= 99 else f"{value} years"
    if param == "nonsolicit_months":
        return f"{value} months"
    if param == "coverage_millions":
        return f"${value:g}M per claim"
    if isinstance(value, bool):
        return {"uncapped": "uncapped liability", "data_breach_carveout": "data breach claims carved out of the cap",
                "mutual": "mutual", "auto_renews": "renews automatically", "disclaimer": "disclaimer present"
                }.get(param, str(value)) if value else {
                "uncapped": "capped", "data_breach_carveout": "no data breach carve-out", "mutual": "one-way",
                "auto_renews": "no automatic renewal", "disclaimer": "no disclaimer of implied warranties"
                }.get(param, str(value))
    return str(value).replace("_", " ")


PARAM_LABEL = {
    "cap_multiple": "Liability cap", "uncapped": "Uncapped liability", "data_breach_carveout": "Data breach carve-out",
    "mutual": "Mutuality", "scope": "Indemnity scope", "tfc_notice_days": "Termination for convenience",
    "cure_days": "Cure period", "payment_days": "Payment terms", "late_fee_pct": "Late interest",
    "auto_renews": "Automatic renewal", "renewal_notice_days": "Non-renewal notice", "law": "Governing law",
    "term_years": "Confidentiality term", "breach_hours": "Breach notice", "subprocessors": "Subprocessor approval",
    "audit": "Audit rights", "disclaimer": "Warranty disclaimer", "warranty_days": "Warranty period",
    "assignment": "Assignment", "nonsolicit_months": "Non-solicit period", "coverage_millions": "Insurance limits",
}


@dataclass
class Analysis:
    id: str
    title: str
    kind: str
    text: str
    clauses: list[dict]
    present: dict[str, list[str]]  # clause type -> clause ids
    params: dict[str, dict[str, dict]]
    issues: list[dict]
    summary: dict
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "title", "kind", "text", "clauses", "present", "params",
                                              "issues", "summary", "meta")}


def _confirmed(ctype: str, clauses: list[Clause], found: dict[str, Evidence]) -> bool:
    """A type counts as present when a parameter was read, or the heading names it and it is not a bare reference."""
    if all(REFERENCE.search(c.text) for c in clauses):
        return False
    if any(e.value is not None or e.start is not None for e in found.values()):
        return True
    from .classify import HEADING_KEYWORDS

    keys = HEADING_KEYWORDS[ctype] + HEADING_HITS.get(ctype, [])
    return any(any(k in f"{c.heading} {c.label}".lower() for k in keys) for c in clauses)


def _redline(chk, ev: Evidence, text: str, clauses: list[Clause], vendor: str) -> dict:
    model = chk.redline.replace("Tessavel", vendor)
    if ev.start is None:
        pos = max(c.end for c in clauses)
        return {"kind": "insert", "start": pos, "end": pos, "original": "", "proposed": " " + model}
    original = text[ev.start:ev.end]
    proposed = model
    if chk.param == "data_breach_carveout":
        trimmed = re.sub(r",?\s*or \(d\)[^.]*", "", original)
        trimmed = re.sub(r",\s*\(c\)", ", or (c)", trimmed) if " or (c)" not in trimmed else trimmed
        proposed = f"{trimmed} {model}"
    return {"kind": "replace", "start": ev.start, "end": ev.end, "original": original, "proposed": proposed}


class Pipeline:
    def __init__(self, playbook: Playbook | None = None, classifier: Classifier | None = None,
                 extractor: Extractor | None = None):
        self.playbook = playbook or load_playbook()
        self.classifier = classifier or BaselineClassifier(self.playbook)
        self.extractor = extractor or Extractor()

    def vendor_alias(self, text: str) -> str:
        """What the customer paper calls the vendor: 'Tessavel, Inc. ("Supplier")' -> Supplier."""
        m = re.search(re.escape(self.playbook.vendor) + r'[^(\n]{0,20}\(\s*(?:the\s+)?["\u201c]([A-Z][A-Za-z]+)["\u201d]', text)
        return m.group(1) if m else self.playbook.vendor

    @property
    def mode(self) -> str:
        return f"{self.classifier.name}+{self.extractor.name}"

    def analyze(self, text: str, doc_id: str = "doc", title: str | None = None) -> Analysis:
        start = time.perf_counter()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        seg: Segmented = segment(text)
        self.classifier.classify(seg)
        groups: dict[str, list[Clause]] = {}
        for c in seg.clauses:
            for t in c.types:
                groups.setdefault(t, []).append(c)
        extracted = self.extractor.extract(seg, text, groups)
        present: dict[str, list[str]] = {}
        params: dict[str, dict[str, dict]] = {}
        for t in TYPES:
            if t in groups and _confirmed(t, groups[t], extracted.get(t, {})):
                present[t] = [c.id for c in groups[t]]
                params[t] = {p: e.to_dict() for p, e in extracted.get(t, {}).items()}
        issues = self._issues(text, seg, groups, present, extracted)
        clause_sev: dict[str, str] = {}
        for it in issues:
            for cid in it["clause_ids"]:
                clause_sev[cid] = worst([clause_sev.get(cid, "ok"), it["severity"]])
        clauses = [{"id": c.id, "number": c.number, "heading": c.heading, "label": c.label, "start": c.start,
                    "end": c.end, "types": [t for t in c.types if t in present],
                    "severity": clause_sev.get(c.id, "ok")} for c in seg.clauses]
        summary = summarize(issues, self.playbook)
        meta = {"mode": self.mode, "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "segments": len(seg.clauses), "definitions": sorted(seg.definitions)}
        return Analysis(doc_id, title or seg.title, seg.kind, text, clauses, present, params, issues, summary, meta)

    def _issues(self, text, seg, groups, present, extracted) -> list[dict]:
        pb = self.playbook
        issues: list[dict] = []
        for t in TYPES:
            rule = pb.rules[t]
            if t not in present:
                if seg.kind in rule.required_in:
                    pos = len(text.rstrip())
                    sev = rule.missing_severity
                    issues.append({
                        "type": t, "label": rule.label, "param": None, "value": None, "severity": sev,
                        "band": "missing", "reason": rule.missing_reason or f"No {rule.label.lower()} clause.",
                        "position": {"preferred": rule.model_clause}, "approvers": pb.approvers_for(sev, rule.missing_approvers),
                        "clause_ids": [], "evidence": None,
                        "redline": {"kind": "insert", "start": pos, "end": pos, "original": "",
                                    "proposed": f"\n\n{rule.label}. "
                                                + rule.model_clause.replace(pb.vendor, self.vendor_alias(text))}})
                continue
            found = extracted.get(t, {})
            for chk in rule.checks:
                ev = found.get(chk.param)
                if ev is None:
                    continue
                sev, band = chk.evaluate(ev.value)
                if sev == "ok":
                    continue
                pos_text = chk.positions.get("preferred", "")
                reason = (f"{PARAM_LABEL.get(chk.param, chk.param)}: {describe(chk.param, ev.value)}. "
                          f"Playbook: {pos_text}"
                          + ("" if band == "fallback" else f" Fallback: {chk.positions.get('fallback', 'none')}"))
                issues.append({
                    "type": t, "label": rule.label, "param": chk.param, "value": ev.value, "severity": sev,
                    "band": band, "reason": reason, "position": chk.positions,
                    "approvers": pb.approvers_for(sev, chk.approvers),
                    "clause_ids": [c.id for c in groups[t]
                                   if ev.start is not None and c.start <= ev.start < c.end + 1] or
                                  [groups[t][-1].id],
                    "evidence": ev.to_dict(), "redline": _redline(chk, ev, text, groups[t], self.vendor_alias(text))})
        issues.sort(key=lambda i: (-RANK[i["severity"]], TYPES.index(i["type"])))
        for n, it in enumerate(issues, 1):
            it["id"] = f"I{n:02d}"
        return issues


def summarize(issues: list[dict], playbook: Playbook) -> dict:
    by = {s: 0 for s in ("high", "medium", "low")}
    for it in issues:
        by[it["severity"]] += 1
    roles = sorted({r for it in issues for r in it["approvers"]})
    # Saturating score: 0 with no issues, about 50 for two high issues, never quite 100.
    score = round(100 * (1 - math.exp(-sum(WEIGHTS[it["severity"]] for it in issues) / 70)))
    return {"risk_score": score, "by_severity": by, "issues": len(issues),
            "missing": [it["type"] for it in issues if it["band"] == "missing"],
            "approvals_needed": [{"role": r, "group": playbook.approver_groups.get(r, r)} for r in roles]}
