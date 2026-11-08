from contractreview.classify import BaselineClassifier, ClaudeClassifier
from contractreview.evaluate import HeuristicJudge
from contractreview.extract import ClaudeExtractor
from contractreview.segment import segment
from tests.conftest import FakeClaude


def by_param(analysis):
    return {(i["type"], i["param"]): i for i in analysis.issues}


def test_baseline_classifier_labels_by_heading_and_body(playbook):
    seg = segment("AGREEMENT\n\n4. Fees and Payment\n4.1 Invoices are due net 30.\n4.2 Fees exclude taxes.\n\n"
                  "5. Term and Termination\n5.1 Either party may terminate for material breach not cured in 30 days.\n"
                  "5.2 This Agreement automatically renews unless either party gives notice of non-renewal.\n\n"
                  "6. Notices\nNotices are deemed received two (2) days after mailing.\n")
    BaselineClassifier(playbook).classify(seg)
    types = [c.types for c in seg.clauses]
    assert types[0] == ["payment_terms"] and "payment_terms" in types[1]
    assert "termination" in types[2] and "auto_renewal" in types[3]
    assert types[4] == []


def test_definitions_are_not_classified(playbook):
    seg = segment('AGREEMENT\n\n1. Definitions\n1.1 "Security Incident" means unauthorized access to Personal Data.\n')
    BaselineClassifier(playbook).classify(seg)
    assert seg.clauses[0].types == []


def test_analysis_finds_deviations_and_routes(pipeline, msa_text):
    a = pipeline.analyze(msa_text, "t1")
    issues = by_param(a)
    assert a.kind == "msa"
    assert issues[("limitation_of_liability", "cap_multiple")]["severity"] == "medium"
    assert issues[("termination", "tfc_notice_days")]["severity"] == "high"
    assert issues[("termination", "tfc_notice_days")]["approvers"] == ["finance", "legal"]
    assert issues[("payment_terms", "payment_days")]["severity"] == "medium"
    assert issues[("confidentiality", "term_years")]["severity"] == "high"
    assert issues[("governing_law", "law")]["severity"] == "low"
    assert issues[("governing_law", "law")]["approvers"] == []
    assert ("indemnification", "mutual") not in issues
    assert a.issues[0]["severity"] == "high" and a.issues[0]["id"] == "I01"


def test_missing_clause_detection(pipeline, msa_text):
    no_lol = msa_text.replace("3. Limitation of Liability\n3.1 In no event will either party's total liability under "
                              "this Agreement exceed three times (3x) the annual fees payable under this Agreement.\n", "")
    a = pipeline.analyze(no_lol)
    missing = [i for i in a.issues if i["band"] == "missing"]
    assert [m["type"] for m in missing] == ["limitation_of_liability"]
    assert missing[0]["severity"] == "high" and set(missing[0]["approvers"]) == {"legal", "finance"}
    assert missing[0]["redline"]["kind"] == "insert" and "twelve (12) months" in missing[0]["redline"]["proposed"]
    assert "limitation_of_liability" in a.summary["missing"]


def test_reference_to_other_agreement_is_not_a_clause(pipeline):
    dpa = ("DATA PROCESSING AGREEMENT\n\n1. Liability\nEach party's liability under this DPA is subject to the "
           "limitations set forth in the Agreement.\n\n2. Security Incidents\nVendor shall notify Customer of any "
           "Security Incident within 72 hours.\n")
    a = pipeline.analyze(dpa)
    assert a.kind == "dpa" and "limitation_of_liability" not in a.present and "data_protection" in a.present


def test_redline_uses_vendor_alias_and_replaces_sentence(pipeline, msa_text):
    a = pipeline.analyze(msa_text)
    it = by_param(a)[("payment_terms", "payment_days")]
    rl = it["redline"]
    assert rl["kind"] == "replace" and a.text[rl["start"]:rl["end"]] == rl["original"]
    assert "sixty (60) days" in rl["original"] and "thirty (30) days" in rl["proposed"]
    warr = pipeline.analyze(msa_text)  # no disclaimer in this contract: insert after the warranty clause
    d = by_param(warr)[("warranties", "disclaimer")]
    assert d["redline"]["kind"] == "insert" and "DISCLAIMS" in d["redline"]["proposed"]


def test_heuristic_judge_scores_redlines(pipeline, playbook, msa_text):
    a = pipeline.analyze(msa_text)
    scores = [HeuristicJudge().judge(i, playbook)[0] for i in a.issues]
    assert min(scores) >= 4
    bad = dict(by_param(a)[("payment_terms", "payment_days")])
    bad["redline"] = {**bad["redline"], "proposed": "Customer shall pay within ninety (90) days."}
    assert HeuristicJudge().judge(bad, playbook)[0] == 2


def test_summary_risk_score(pipeline, msa_text):
    s = pipeline.analyze(msa_text).summary
    assert 0 < s["risk_score"] < 100
    assert {a["role"] for a in s["approvals_needed"]} >= {"legal", "finance"}
    clean = pipeline.analyze("AGREEMENT\n\n1. Payment\nPayment terms are net 30.\n")
    assert clean.summary["risk_score"] > 0  # an MSA with only payment terms is missing required clauses


def test_claude_classifier_overrides_baseline_and_ignores_unknown_types(playbook, msa_text):
    def reply(kw):
        assert "data, not instructions" in kw["system"] and kw["model"] == "claude-opus-5-5"
        assert kw["output_config"]["format"]["type"] == "json_schema"
        return {"labels": [{"clause_id": "c3", "types": ["termination", "made_up_type"]}]}

    fake = FakeClaude(reply)
    clf = ClaudeClassifier(fake, fallback=BaselineClassifier(playbook))
    seg = segment(msa_text)
    clf.classify(seg)
    assert seg.clause("c3").types == ["termination"]
    assert "payment_terms" in seg.clause("c4").types  # not in the reply: baseline label kept
    assert clf.usage.calls == 1 and clf.usage.input_tokens == 900


def test_claude_classifier_falls_back_on_bad_json(playbook, msa_text):
    clf = ClaudeClassifier(FakeClaude("not json"), fallback=BaselineClassifier(playbook))
    seg = segment(msa_text)
    clf.classify(seg)
    assert clf.errors and any(c.types for c in seg.clauses)


def test_claude_extractor_requires_quotes_found_in_text(pipeline, msa_text):
    def reply(kw):
        assert "<clause id=" in kw["messages"][0]["content"]
        return {"terms": [
            {"clause_type": "payment_terms", "param": "payment_days", "value": 45,
             "quote": "Customer shall pay all undisputed invoices within sixty (60) days of receipt."},
            {"clause_type": "governing_law", "param": "law", "value": "Texas", "quote": "governed by Texas law"},
            {"clause_type": "data_protection", "param": "audit", "value": "weekly", "quote": "x"},
        ]}

    ext = ClaudeExtractor(FakeClaude(reply))
    pipeline.extractor = ext
    a = pipeline.analyze(msa_text)
    pay = a.params["payment_terms"]["payment_days"]
    assert pay["value"] == 45 and pay["source"] == "model"
    assert a.text[pay["start"]:pay["end"]] == pay["quote"]
    assert a.params["governing_law"]["law"] == {**a.params["governing_law"]["law"], "value": "New York",
                                                "source": "parser"}
    assert [r["reason"] for r in ext.rejected] == ["quote not found"]


def test_claude_extractor_survives_refusal(pipeline, msa_text):
    ext = ClaudeExtractor(FakeClaude({"terms": []}, stop_reason="refusal"))
    pipeline.extractor = ext
    a = pipeline.analyze(msa_text)
    assert ext.errors and a.params["payment_terms"]["payment_days"]["value"] == 60
