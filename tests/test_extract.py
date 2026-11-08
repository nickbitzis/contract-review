import pytest

from contractreview.extract import durations, locate, parse_type, valid_value
from contractreview.segment import Clause


def parse(ctype, text, defs=None):
    found = parse_type(ctype, [Clause("c1", "", "", text, 0, len(text))], defs or {})
    return {k: v.value for k, v in found.items()}


def test_durations_words_and_digits():
    assert durations("within thirty (30) days and 72 hours, or one (1) business day") == [
        (30, "day"), (72, "hour"), (1, "business_day")]


@pytest.mark.parametrize("text,expected", [
    ("Each party's aggregate liability shall not exceed the fees paid in the twenty-four (24) months preceding the claim.", 2.0),
    ("In no event will total liability exceed two times (2x) the annual fees.", 2.0),
    ("Total liability shall not exceed fifty percent (50%) of the annual fees.", 0.5),
    ("Liability shall not exceed the fees paid in the twelve (12) months before the claim.", 1.0),
])
def test_cap_multiple(text, expected):
    assert parse("limitation_of_liability", text)["cap_multiple"] == expected


def test_cap_from_defined_term():
    defs = {"Liability Cap": "an amount equal to the fees paid in the thirty-six (36) months before the claim."}
    got = parse("limitation_of_liability", "Each party's total liability is limited to the Liability Cap.", defs)
    assert got["cap_multiple"] == 3.0


def test_uncapped_and_data_breach_carveout():
    assert parse("limitation_of_liability", "Supplier's liability under this Agreement shall be unlimited.")["uncapped"]
    got = parse("limitation_of_liability", "Liability shall not exceed the fees paid in the 12 months before the claim. "
                "The limitations in Section 9 do not apply to Supplier's breach of its data protection obligations.")
    assert got == {"cap_multiple": 1.0, "uncapped": False, "data_breach_carveout": True}


@pytest.mark.parametrize("text,expected", [
    ("Customer may terminate this Agreement for convenience on sixty (60) days' prior written notice.", 60),
    ("Customer may end this Agreement without cause on 30 days' notice.", 30),
    ("Neither party may terminate this Agreement for convenience.", None),
])
def test_termination_for_convenience(text, expected):
    got = parse("termination", text)
    assert "tfc_notice_days" in got and got["tfc_notice_days"] == expected


def test_cure_period():
    assert parse("termination", "Either party may terminate if the other fails to remedy such breach within 10 days "
                 "of written notice.")["cure_days"] == 10


@pytest.mark.parametrize("text,days,fee", [
    ("Payment terms are net 45 days from the invoice date. Late payments bear interest at 1% per month.", 45, 1.0),
    ("Customer will remit payment no later than the sixtieth day following the invoice date.", 60, None),
    ("Customer shall pay all undisputed invoices within ninety (90) days of receipt. No late fees shall apply.", 90, 0.0),
])
def test_payment(text, days, fee):
    got = parse("payment_terms", text)
    assert got["payment_days"] == days and got.get("late_fee_pct") == fee


def test_renewal():
    got = parse("auto_renewal", "This Agreement automatically renews for one (1) year periods unless either party gives "
                "notice of non-renewal at least forty-five (45) days before the end of the term.")
    assert got == {"auto_renews": True, "renewal_notice_days": 45}
    assert parse("auto_renewal", "The Subscription Term will not renew automatically.")["auto_renews"] is False
    assert parse("auto_renewal", "Renewal: Automatic, 30 days notice of non-renewal")["renewal_notice_days"] == 30


@pytest.mark.parametrize("text,law", [
    ("This Agreement is governed by the laws of the State of Delaware.", "Delaware"),
    ("This Agreement shall be construed in accordance with the laws of England and Wales.", "England and Wales"),
    ("This Agreement is governed by the laws of the Province of Ontario.", "Ontario"),
])
def test_governing_law(text, law):
    assert parse("governing_law", text)["law"] == law


def test_confidentiality_term():
    assert parse("confidentiality", "These obligations continue for five (5) years after termination.")["term_years"] == 5
    assert parse("confidentiality", "These obligations survive in perpetuity.")["term_years"] == 99


@pytest.mark.parametrize("text,param,value", [
    ("Vendor shall notify Customer of any Security Incident within twenty-four (24) hours.", "breach_hours", 24),
    ("Vendor shall notify Customer of any Security Incident within two (2) business days.", "breach_hours", 48),
    ("Vendor shall notify Customer of any Security Incident without undue delay.", "breach_hours", None),
    ("Vendor shall not engage any subprocessor without Customer's prior written consent.", "subprocessors", "consent"),
    ("Vendor will notify Customer before engaging a new subprocessor, and Customer may object.", "subprocessors",
     "objection"),
    ("Customer may inspect Vendor's facilities on-site as often as it deems necessary.", "audit", "onsite"),
    ("Vendor will provide its most recent SOC 2 Type II report once per year.", "audit", "report"),
    ("Customer may audit Vendor's compliance once per year on thirty (30) days' notice.", "audit", "annual"),
])
def test_data_protection(text, param, value):
    assert parse("data_protection", text)[param] == value


def test_warranties():
    got = parse("warranties", "Vendor warrants the Services will perform for six (6) months.")
    assert got == {"warranty_days": 180, "disclaimer": False}
    got = parse("warranties", "Vendor warrants performance for one (1) year. VENDOR DISCLAIMS ALL IMPLIED WARRANTIES.")
    assert got == {"warranty_days": 365, "disclaimer": True}


@pytest.mark.parametrize("text,value", [
    ("Neither party may assign this Agreement without consent, except to a successor in a merger.", "mutual_coc"),
    ("Neither party may assign this Agreement without the prior written consent of the other party.", "mutual_consent"),
    ("Vendor may not assign this Agreement without consent. Customer may assign this Agreement freely.", "one_sided"),
])
def test_assignment(text, value):
    assert parse("assignment", text)["assignment"] == value


def test_non_solicit_and_insurance():
    assert parse("non_solicit", "Neither party shall solicit any employee for eighteen (18) months.") == {
        "nonsolicit_months": 18}
    assert parse("insurance", "Vendor shall maintain insurance of not less than $5,000,000 per occurrence.") == {
        "coverage_millions": 5}
    assert parse("insurance", "Vendor shall maintain cyber insurance of $10 million.") == {"coverage_millions": 10}


def test_evidence_spans_point_at_the_sentence():
    text = "Intro sentence. Payment terms are net 60."
    found = parse_type("payment_terms", [Clause("c1", "", "", text, 500, 500 + len(text))], {})
    ev = found["payment_days"]
    assert (ev.start, ev.end) == (516, 541) and ev.quote == "Payment terms are net 60."


def test_locate_and_value_validation():
    assert locate("a  b\nc d", "b c") == (3, 6)
    assert locate("abc", "zzz") is None
    assert valid_value("audit", "onsite") and not valid_value("audit", "weekly")
    assert not valid_value("payment_days", True) and valid_value("payment_days", 30)
