"""The negotiation playbook: load the YAML, compare extracted parameters to it, and route approvals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).parent / "playbook.yaml"
SEVERITIES = ("ok", "low", "medium", "high")
RANK = {s: i for i, s in enumerate(SEVERITIES)}


@dataclass
class Check:
    clause_type: str
    param: str
    kind: str  # max | min | enum | flag
    approvers: list[str]
    positions: dict[str, str]
    redline: str
    preferred: Any = None
    fallback: Any = None
    walkaway: Any = None
    otherwise: str = "medium"
    bad: Any = None
    severity: str = "medium"

    def evaluate(self, value: Any) -> tuple[str, str]:
        """Return (severity, band) for one extracted value. None means the term is absent: nothing to flag."""
        if value is None:
            return "ok", "absent"
        if self.kind == "flag":
            return (self.severity, "walkaway") if value == self.bad else ("ok", "preferred")
        if self.kind == "enum":
            if value in (self.preferred or []):
                return "ok", "preferred"
            if value in (self.fallback or []):
                return "low", "fallback"
            if value in (self.walkaway or []):
                return "high", "walkaway"
            return self.otherwise, "outside"
        v = float(value)
        if self.kind == "max":
            bands = [(v <= self.preferred, "ok", "preferred"), (v <= self.fallback, "low", "fallback"),
                     (self.walkaway is None or v <= self.walkaway, "medium", "outside")]
        elif self.kind == "min":
            bands = [(v >= self.preferred, "ok", "preferred"), (v >= self.fallback, "low", "fallback"),
                     (self.walkaway is None or v >= self.walkaway, "medium", "outside")]
        else:
            raise ValueError(f"unknown check kind {self.kind!r}")
        for cond, sev, band in bands:
            if cond:
                return sev, band
        return "high", "walkaway"


@dataclass
class ClauseRule:
    clause_type: str
    label: str
    required_in: list[str]
    model_clause: str
    checks: list[Check] = field(default_factory=list)
    missing_severity: str = "medium"
    missing_approvers: list[str] = field(default_factory=list)
    missing_reason: str = ""


@dataclass
class Playbook:
    vendor: str
    rules: dict[str, ClauseRule]
    approver_groups: dict[str, str]
    escalation: dict[str, list[str]]

    def approvers_for(self, severity: str, base: list[str]) -> list[str]:
        """Approvers are needed only outside the fallback position (medium and high). High adds the escalation."""
        if RANK[severity] < RANK["medium"]:
            return []
        roles = list(base)
        for r in self.escalation.get(severity, []):
            if r not in roles:
                roles.append(r)
        return roles

    def required(self, doc_kind: str) -> list[str]:
        return [t for t, r in self.rules.items() if doc_kind in r.required_in]

    def to_dict(self) -> dict:
        return {
            "vendor": self.vendor,
            "approver_groups": self.approver_groups,
            "clauses": {t: {"label": r.label, "required_in": r.required_in, "model_clause": r.model_clause,
                            "checks": [{"param": c.param, "kind": c.kind, "positions": c.positions,
                                        "approvers": c.approvers} for c in r.checks]}
                        for t, r in self.rules.items()},
        }


def load_playbook(path: Path | str | None = None) -> Playbook:
    raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text())
    rules: dict[str, ClauseRule] = {}
    for ctype, spec in raw["clauses"].items():
        checks = [Check(clause_type=ctype, param=c["param"], kind=c["kind"], approvers=list(c.get("approvers", [])),
                        positions=dict(c.get("positions", {})), redline=" ".join(c["redline"].split()),
                        preferred=c.get("preferred"), fallback=c.get("fallback"), walkaway=c.get("walkaway"),
                        otherwise=c.get("otherwise", "medium"), bad=c.get("bad"), severity=c.get("severity", "medium"))
                  for c in spec.get("checks", [])]
        for c in checks:
            if c.kind not in ("max", "min", "enum", "flag"):
                raise ValueError(f"{ctype}.{c.param}: unknown kind {c.kind}")
        rules[ctype] = ClauseRule(ctype, spec["label"], list(spec.get("required_in", [])),
                                  " ".join(spec["model_clause"].split()), checks,
                                  spec.get("missing_severity", "medium"), list(spec.get("missing_approvers", [])),
                                  spec.get("missing_reason", ""))
    return Playbook(raw.get("vendor", "Vendor"), rules, dict(raw.get("approver_groups", {})),
                    {k: list(v) for k, v in (raw.get("escalation") or {}).items()})


def worst(severities) -> str:
    return max(severities, key=lambda s: RANK[s], default="ok")
