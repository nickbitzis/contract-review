from collections import Counter

from contractreview.corpus import build_corpus, write_corpus
from contractreview.evaluate import ClaudeJudge, _pr, gate, markdown_table, run_eval, same
from tests.conftest import FakeClaude


def test_corpus_is_seeded_and_labeled(corpus):
    again = build_corpus()
    assert len(corpus) == 40 and [c.text for c in corpus] == [c.text for c in again]
    assert Counter(c.kind for c in corpus) == {"msa": 24, "dpa": 8, "order_form": 8}
    assert any(m["type"] == "limitation_of_liability" for c in corpus for m in c.missing)
    phr = Counter(cl.phrasing for c in corpus for cl in c.clauses.values())
    assert phr["unusual"] > 40 and phr["standard"] > 100
    for c in corpus:
        assert "Tessavel" in c.text and "\u2014" not in c.text
        for d in c.deviations:
            assert d["severity"] in ("low", "medium", "high")


def test_write_corpus(tmp_path):
    paths = write_corpus(tmp_path)
    assert len(paths) == 41 and (tmp_path / "truth.jsonl").read_text().count("\n") == 40


def test_metric_helpers():
    assert _pr(8, 2, 0) == {"precision": 0.8, "recall": 1.0, "tp": 8, "fp": 2, "fn": 0}
    assert _pr(0, 0, 0)["recall"] == 1.0
    assert same(None, False) and same(2, 2.0) and not same(True, None) and not same("a", "b")


def test_baseline_eval_and_gate(corpus):
    r = run_eval("baseline", corpus)
    assert r.contracts == 40
    assert r.detection_micro["precision"] > 0.9 and r.detection_micro["recall"] > 0.9
    assert r.param_by_phrasing["standard"] > r.param_by_phrasing["unusual"]
    assert r.high["recall"] >= 0.8 and r.missing["recall"] == 1.0
    assert gate(r, 0.8) == [] and gate(r, 0.999)
    assert "High-severity deviation recall" in markdown_table([r])


def test_simulated_model_mode_improves_recall(corpus):
    base = run_eval("baseline", corpus[:12])
    sim = run_eval("simulated-model", corpus[:12])
    assert sim.model_calls == 24
    assert sim.param_accuracy > base.param_accuracy
    assert sim.deviation["recall"] >= base.deviation["recall"]


def test_claude_judge_with_fake_client(playbook):
    it = {"reason": "Payment terms: net 60.", "position": {"preferred": "Net 30."},
          "redline": {"original": "net 60", "proposed": "net 30"}}
    assert ClaudeJudge(FakeClaude({"score": 9, "rationale": "fine"})).judge(it, playbook) == (5, "fine")
    assert ClaudeJudge(FakeClaude("oops")).judge(it, playbook)[0] == 0
