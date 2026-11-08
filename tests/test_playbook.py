import pytest

from contractreview.classify import TYPES
from contractreview.playbook import Check, load_playbook


def check(pb, ctype, param):
    return next(c for c in pb.rules[ctype].checks if c.param == param)


def test_playbook_covers_every_clause_type(playbook):
    assert set(playbook.rules) == set(TYPES)
    for r in playbook.rules.values():
        assert r.model_clause and all(c.redline and c.positions.get("preferred") for c in r.checks)


@pytest.mark.parametrize("ctype,param,value,severity", [
    ("limitation_of_liability", "cap_multiple", 1.0, "ok"),
    ("limitation_of_liability", "cap_multiple", 2.0, "low"),
    ("limitation_of_liability", "cap_multiple", 3.0, "medium"),
    ("limitation_of_liability", "cap_multiple", 5.0, "high"),
    ("limitation_of_liability", "uncapped", True, "high"),
    ("limitation_of_liability", "data_breach_carveout", True, "medium"),
    ("indemnification", "mutual", False, "medium"),
    ("indemnification", "scope", "ip_and_data", "low"),
    ("indemnification", "scope", "any_claim", "high"),
    ("termination", "tfc_notice_days", None, "ok"),
    ("termination", "tfc_notice_days", 90, "low"),
    ("termination", "tfc_notice_days", 30, "medium"),
    ("termination", "tfc_notice_days", 15, "high"),
    ("termination", "cure_days", 10, "medium"),
    ("payment_terms", "payment_days", 30, "ok"),
    ("payment_terms", "payment_days", 45, "low"),
    ("payment_terms", "payment_days", 60, "medium"),
    ("payment_terms", "payment_days", 90, "high"),
    ("payment_terms", "late_fee_pct", 0.0, "medium"),
    ("auto_renewal", "auto_renews", False, "low"),
    ("auto_renewal", "renewal_notice_days", 10, "high"),
    ("governing_law", "law", "Delaware", "ok"),
    ("governing_law", "law", "California", "low"),
    ("governing_law", "law", "Ontario", "medium"),
    ("confidentiality", "term_years", 99, "high"),
    ("data_protection", "breach_hours", 48, "low"),
    ("data_protection", "breach_hours", 12, "high"),
    ("data_protection", "subprocessors", "consent", "high"),
    ("data_protection", "audit", "annual", "low"),
    ("warranties", "disclaimer", False, "medium"),
    ("warranties", "warranty_days", 365, "medium"),
    ("assignment", "assignment", "one_sided", "medium"),
    ("non_solicit", "nonsolicit_months", 36, "high"),
    ("insurance", "coverage_millions", 20, "high"),
])
def test_comparison_per_clause_type(playbook, ctype, param, value, severity):
    assert check(playbook, ctype, param).evaluate(value)[0] == severity


def test_approver_routing(playbook):
    assert playbook.approvers_for("low", ["finance"]) == []
    assert playbook.approvers_for("medium", ["finance"]) == ["finance"]
    assert playbook.approvers_for("high", ["security"]) == ["security", "legal"]
    assert playbook.approvers_for("high", ["legal", "finance"]) == ["legal", "finance"]


def test_required_by_document_kind(playbook):
    assert "limitation_of_liability" in playbook.required("msa")
    assert playbook.required("dpa") == ["data_protection"]
    assert playbook.required("order_form") == ["payment_terms"]


def test_unknown_check_kind_rejected(tmp_path):
    p = tmp_path / "pb.yaml"
    p.write_text("clauses:\n  x:\n    label: X\n    model_clause: y\n    checks:\n      - {param: a, kind: weird, "
                 "redline: z}\n")
    with pytest.raises(ValueError):
        load_playbook(p)


def test_enum_otherwise():
    c = Check("t", "p", "enum", [], {}, "r", preferred=["a"], fallback=["b"], walkaway=["c"], otherwise="medium")
    assert [c.evaluate(v)[0] for v in ("a", "b", "c", "d")] == ["ok", "low", "high", "medium"]
