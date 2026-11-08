"""Review workflow: decisions on each suggested redline, approval routing, the audit trail, and exports.

Decisions:
- accept: send the playbook redline as proposed. It restores a playbook position, so no approval is needed.
- edit: send the reviewer's own wording. The edit is re-read with the same parser and scored again. If it still
  sits outside the fallback position (or cannot be read), the issue's approvers must sign off.
- reject: keep the customer's language. Conceding a medium or high issue needs the approvers on the issue.

An approver can only approve for a role the issue is waiting on. The contract is ready to sign when every issue
has a decision and every required approval is in. Every action is appended to the audit trail.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime

from .analyze import PARAM_LABEL, Analysis, Pipeline, describe
from .extract import parse_type
from .playbook import Playbook
from .redline import final_text, redlined_html
from .segment import Clause

DECISIONS = ("accept", "edit", "reject")


class ReviewError(ValueError):
    pass


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def rescore_edit(issue: dict, edited: str, playbook: Playbook) -> tuple[str, object]:
    """Severity of the reviewer's wording for the same parameter. Unreadable wording keeps the original severity."""
    if issue["param"] is None:
        return ("ok", None) if parse_type(issue["type"], [Clause("e", "", "", edited, 0, len(edited))], {}) \
            else (issue["severity"], None)
    chk = next(c for c in playbook.rules[issue["type"]].checks if c.param == issue["param"])
    found = parse_type(issue["type"], [Clause("e", "", "", edited, 0, len(edited))], {})
    ev = found.get(chk.param)
    if ev is None:
        return (("ok", None) if chk.param == "tfc_notice_days" else (issue["severity"], None))
    return chk.evaluate(ev.value)[0], ev.value


class Store:
    """In-memory store for contracts under review. One lock, safe for the single-process demo server."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.playbook = pipeline.playbook
        self.items: dict[str, dict] = {}
        self.lock = threading.Lock()

    def create(self, text: str, title: str | None = None, source: str = "upload", actor: str = "system") -> dict:
        cid = uuid.uuid4().hex[:10]
        a: Analysis = self.pipeline.analyze(text, cid, title)
        doc = a.to_dict()
        doc["source"] = source
        for it in doc["issues"]:
            it.update({"status": "pending", "decision_text": None, "required_approvals": [], "approvals": {},
                       "edit_severity": None, "note": ""})
        doc["audit"] = []
        with self.lock:
            self.items[cid] = doc
            self._log(doc, actor, "analyzed", None, f"{len(doc['issues'])} issues, risk {doc['summary']['risk_score']}, "
                                                    f"mode {doc['meta']['mode']}")
            self._refresh(doc)
        return doc

    def get(self, cid: str) -> dict:
        if cid not in self.items:
            raise KeyError(cid)
        return self.items[cid]

    def list(self) -> list[dict]:
        return [{"id": d["id"], "title": d["title"], "kind": d["kind"], "status": d["status"],
                 "risk_score": d["summary"]["risk_score"]} for d in self.items.values()]

    def _issue(self, doc: dict, issue_id: str) -> dict:
        for it in doc["issues"]:
            if it["id"] == issue_id:
                return it
        raise KeyError(issue_id)

    def _log(self, doc: dict, actor: str, action: str, issue_id: str | None, detail: str = "") -> None:
        doc["audit"].append({"seq": len(doc["audit"]) + 1, "at": _now(), "actor": actor, "action": action,
                             "issue_id": issue_id, "detail": detail})

    def decide(self, cid: str, issue_id: str, decision: str, reviewer: str, text: str | None = None,
               note: str = "") -> dict:
        if decision not in DECISIONS:
            raise ReviewError(f"decision must be one of {', '.join(DECISIONS)}")
        with self.lock:
            doc = self.get(cid)
            it = self._issue(doc, issue_id)
            it["approvals"], it["note"], it["edit_severity"] = {}, note, None
            if decision == "accept":
                it["status"], it["decision_text"], it["required_approvals"] = "accepted", it["redline"]["proposed"], []
                detail = "redline accepted as proposed"
            elif decision == "edit":
                if not text or not text.strip():
                    raise ReviewError("an edit needs text")
                sev, val = rescore_edit(it, text, self.playbook)
                it["status"], it["decision_text"], it["edit_severity"] = "edited", text, sev
                it["required_approvals"] = self.playbook.approvers_for(sev, self._base_approvers(it))
                detail = f"edited; edit reads as {describe(it['param'], val) if it['param'] else 'clause text'}, " \
                         f"severity {sev}"
            else:
                it["status"], it["decision_text"] = "rejected", None
                it["required_approvals"] = list(it["approvers"])
                detail = "customer language kept" + (f": {note}" if note else "")
            self._log(doc, reviewer, decision, issue_id, detail)
            if it["required_approvals"]:
                self._log(doc, "system", "routed", issue_id, "approval needed from " + ", ".join(
                    self.playbook.approver_groups.get(r, r) for r in it["required_approvals"]))
            self._refresh(doc)
            return it

    def _base_approvers(self, it: dict) -> list[str]:
        if it["param"] is None:
            return self.playbook.rules[it["type"]].missing_approvers
        return next(c.approvers for c in self.playbook.rules[it["type"]].checks if c.param == it["param"])

    def approve(self, cid: str, issue_id: str, role: str, approver: str, approve: bool = True, note: str = "") -> dict:
        with self.lock:
            doc = self.get(cid)
            it = self._issue(doc, issue_id)
            if role not in it["required_approvals"]:
                raise ReviewError(f"issue {issue_id} is not waiting on {role}")
            if role in it["approvals"]:
                raise ReviewError(f"{role} already signed off on {issue_id}")
            if approve:
                it["approvals"][role] = {"by": approver, "at": _now()}
                self._log(doc, approver, "approved", issue_id, f"as {role}" + (f": {note}" if note else ""))
            else:
                it["status"], it["decision_text"], it["required_approvals"], it["approvals"] = "pending", None, [], {}
                self._log(doc, approver, "declined", issue_id, f"as {role}, back to reviewer" +
                          (f": {note}" if note else ""))
            self._refresh(doc)
            return it

    def _refresh(self, doc: dict) -> None:
        issues = doc["issues"]
        pending = [i for i in issues if i["status"] == "pending"]
        waiting = [(i["id"], r) for i in issues for r in i["required_approvals"] if r not in i["approvals"]]
        doc["status"] = "in_review" if pending else "awaiting_approval" if waiting else "ready_to_sign"
        doc["progress"] = {"decided": len(issues) - len(pending), "total": len(issues),
                           "approvals_waiting": [{"issue_id": i, "role": r,
                                                  "group": self.playbook.approver_groups.get(r, r)} for i, r in waiting]}

    def changes(self, doc: dict, include_pending: bool = False) -> list[dict]:
        out = []
        for it in doc["issues"]:
            rl = it["redline"]
            if it["status"] in ("accepted", "edited") or (include_pending and it["status"] == "pending"):
                txt = it["decision_text"] if it["status"] in ("accepted", "edited") else rl["proposed"]
                if rl["kind"] == "insert" and it["status"] == "edited" and not txt.startswith((" ", "\n")):
                    txt = " " + txt
                out.append({"start": rl["start"], "end": rl["end"], "text": txt, "issue_id": it["id"]})
        return out

    def redline(self, cid: str, include_pending: bool = False) -> str:
        doc = self.get(cid)
        note = "Pending suggestions included." if include_pending else "Accepted and edited redlines only."
        return redlined_html(doc["title"], doc["text"], self.changes(doc, include_pending), note)

    def clean_text(self, cid: str) -> str:
        doc = self.get(cid)
        return final_text(doc["text"], self.changes(doc))

    def memo(self, cid: str) -> str:
        return review_memo(self.get(cid), self.playbook)


def review_memo(doc: dict, playbook: Playbook) -> str:
    s = doc["summary"]
    lines = [f"# Contract review memo: {doc['title']}", "",
             "Review aid generated against the Tessavel negotiation playbook. Not legal advice.", "",
             f"- Document type: {doc['kind'].replace('_', ' ')}",
             f"- Status: {doc['status'].replace('_', ' ')}",
             f"- Risk score: {s['risk_score']} / 100",
             f"- Issues: {s['by_severity']['high']} high, {s['by_severity']['medium']} medium, "
             f"{s['by_severity']['low']} low",
             f"- Missing clauses: {', '.join(playbook.rules[t].label for t in s['missing']) or 'none'}",
             f"- Analysis mode: {doc['meta']['mode']}", "",
             "## Issues", "",
             "| ID | Severity | Clause | Term | Customer position | Decision | Approvals |",
             "|---|---|---|---|---|---|---|"]
    for it in doc["issues"]:
        term = PARAM_LABEL.get(it["param"], "Missing clause") if it["param"] else "Missing clause"
        val = describe(it["param"], it["value"]) if it["param"] else "absent"
        appr = ", ".join(f"{playbook.approver_groups.get(r, r)}: "
                         f"{it['approvals'][r]['by'] if r in it['approvals'] else 'waiting'}"
                         for r in it["required_approvals"]) or "none needed"
        lines.append(f"| {it['id']} | {it['severity']} | {it['label']} | {term} | {val} | {it['status']} | {appr} |")
    lines += ["", "## Redlines sent", ""]
    sent = [i for i in doc["issues"] if i["status"] in ("accepted", "edited")]
    for it in sent:
        lines += [f"**{it['id']} {it['label']}** ({it['status']}). {it['reason']}", "",
                  f"> {it['decision_text'].strip()}", ""]
    if not sent:
        lines += ["None yet.", ""]
    conceded = [i for i in doc["issues"] if i["status"] == "rejected"]
    if conceded:
        lines += ["## Customer positions accepted", ""]
        lines += [f"- {i['id']} {i['label']}: {describe(i['param'], i['value']) if i['param'] else 'clause absent'}"
                  + (f". Note: {i['note']}" if i["note"] else "") for i in conceded]
        lines.append("")
    lines += ["## Audit trail", "", "| # | Time (UTC) | Actor | Action | Issue | Detail |", "|---|---|---|---|---|---|"]
    lines += [f"| {e['seq']} | {e['at']} | {e['actor']} | {e['action']} | {e['issue_id'] or ''} | {e['detail']} |"
              for e in doc["audit"]]
    return "\n".join(lines) + "\n"
