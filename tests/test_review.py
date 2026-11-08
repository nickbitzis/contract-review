import pytest

from contractreview.redline import apply_changes, final_text, redlined_html, word_diff
from contractreview.review import ReviewError, Store


@pytest.fixture
def store(pipeline):
    return Store(pipeline)


def issue(doc, ctype, param):
    return next(i for i in doc["issues"] if i["type"] == ctype and i["param"] == param)


def test_accept_needs_no_approval(store, msa_text):
    doc = store.create(msa_text)
    it = issue(doc, "termination", "tfc_notice_days")
    out = store.decide(doc["id"], it["id"], "accept", "ana")
    assert out["status"] == "accepted" and out["required_approvals"] == []


def test_reject_routes_to_issue_approvers(store, msa_text):
    doc = store.create(msa_text)
    it = issue(doc, "termination", "tfc_notice_days")
    out = store.decide(doc["id"], it["id"], "reject", "ana", note="strategic logo")
    assert out["required_approvals"] == ["finance", "legal"]
    assert store.get(doc["id"])["status"] == "in_review"
    with pytest.raises(ReviewError):
        store.approve(doc["id"], it["id"], "security", "sam")
    store.approve(doc["id"], it["id"], "finance", "fay")
    with pytest.raises(ReviewError):
        store.approve(doc["id"], it["id"], "finance", "fay")
    store.approve(doc["id"], it["id"], "legal", "lee")
    assert set(store.get(doc["id"])["issues"][0]["approvals"]) == {"finance", "legal"}


def test_edit_is_rescored(store, msa_text):
    doc = store.create(msa_text)
    it = issue(doc, "termination", "tfc_notice_days")
    ok = store.decide(doc["id"], it["id"], "edit", "ana",
                      "Customer may terminate for convenience on ninety (90) days' prior written notice.")
    assert ok["edit_severity"] == "low" and ok["required_approvals"] == []
    bad = store.decide(doc["id"], it["id"], "edit", "ana",
                       "Customer may terminate for convenience on twenty (20) days' prior written notice.")
    assert bad["edit_severity"] == "high" and bad["required_approvals"] == ["finance", "legal"]
    with pytest.raises(ReviewError):
        store.decide(doc["id"], it["id"], "edit", "ana", "  ")


def test_decline_sends_back_and_status_reaches_ready(store, msa_text):
    doc = store.create(msa_text)
    cid = doc["id"]
    first = doc["issues"][0]
    store.decide(cid, first["id"], "reject", "ana")
    role = store.get(cid)["issues"][0]["required_approvals"][0]
    store.approve(cid, first["id"], role, "x", approve=False)
    assert store.get(cid)["issues"][0]["status"] == "pending"
    for it in store.get(cid)["issues"]:
        store.decide(cid, it["id"], "accept", "ana")
    d = store.get(cid)
    assert d["status"] == "ready_to_sign" and d["progress"]["decided"] == d["progress"]["total"]
    actions = [e["action"] for e in d["audit"]]
    assert actions[:4] == ["analyzed", "reject", "routed", "declined"] and actions.count("accept") == len(d["issues"])


def test_bad_decision_and_unknown_issue(store, msa_text):
    doc = store.create(msa_text)
    with pytest.raises(ReviewError):
        store.decide(doc["id"], doc["issues"][0]["id"], "maybe", "ana")
    with pytest.raises(KeyError):
        store.decide(doc["id"], "I99", "accept", "ana")


def test_memo_and_redline_exports(store, msa_text):
    doc = store.create(msa_text)
    cid = doc["id"]
    pay = issue(doc, "payment_terms", "payment_days")
    store.decide(cid, pay["id"], "accept", "ana")
    tfc = issue(doc, "termination", "tfc_notice_days")
    store.decide(cid, tfc["id"], "reject", "ana")
    memo = store.memo(cid)
    assert "Not legal advice" in memo and "| " + pay["id"] + " |" in memo
    assert "Customer positions accepted" in memo and "Audit trail" in memo
    html = store.redline(cid)
    assert "<del>" in html and "<ins>" in html and "1 change(s)" in html
    assert "thirty (30) days of the invoice date" in store.clean_text(cid)
    assert "sixty (60) days" not in store.clean_text(cid)
    assert "change(s)" in store.redline(cid, include_pending=True)


def test_word_diff_and_overlaps():
    d = word_diff("pay within sixty days", "pay within thirty days")
    assert "<del>sixty</del>" in d and "<ins>thirty</ins>" in d and d.startswith("pay within")
    assert word_diff("abc", "totally different words here").startswith("<del>abc</del>")
    applied, skipped = apply_changes("x" * 30, [{"start": 0, "end": 10, "text": "a"}, {"start": 5, "end": 12, "text": "b"},
                                                 {"start": 30, "end": 30, "text": "c"}])
    assert len(applied) == 2 and len(skipped) == 1
    assert final_text("hello world", [{"start": 6, "end": 11, "text": "there"}]) == "hello there"
    assert "&lt;script&gt;" in redlined_html("<script>", "a <b>", [])
