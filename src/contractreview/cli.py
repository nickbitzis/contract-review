"""Command line: generate the corpus, analyze a file, run the eval and the CI gate, serve the API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_gen(a) -> int:
    from .corpus import write_corpus

    paths = write_corpus(Path(a.out))
    print(f"wrote {len(paths) - 1} contracts and truth.jsonl under {a.out}")
    return 0


def cmd_analyze(a) -> int:
    from .analyze import Pipeline
    from .docx_io import docx_to_text

    path = Path(a.file)
    text = docx_to_text(path.read_bytes()) if path.suffix.lower() == ".docx" else path.read_text()
    res = Pipeline().analyze(text, path.stem)
    if a.json:
        print(json.dumps({k: v for k, v in res.to_dict().items() if k != "text"}, indent=1))
        return 0
    s = res.summary
    print(f"{res.title} ({res.kind}): risk {s['risk_score']}/100, {s['by_severity']}")
    for it in res.issues:
        appr = ",".join(it["approvers"]) or "-"
        print(f"  {it['id']} {it['severity']:<6} {it['label']:<28} {appr:<16} {it['reason'][:90]}")
    return 0


def cmd_eval(a) -> int:
    from .evaluate import gate, markdown_table, per_type_table, run_eval

    modes = ["baseline", "simulated-model"] if a.compare else [a.mode]
    reports = [run_eval(m) for m in modes]
    table = markdown_table(reports)
    print(table)
    print(per_type_table(reports[0]))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([{**r.row(), "failures": r.failures} for r in reports], indent=1))
    if a.summary:
        with open(a.summary, "a") as f:
            f.write("## Contract review eval\n\n" + table + "\n")
    if a.gate:
        target = reports[0]
        problems = gate(target, a.min_high_recall, a.min_recall)
        for p in problems:
            print(f"GATE FAIL ({target.mode}): {p}", file=sys.stderr)
        if problems:
            return 1
        print(f"gate passed for {target.mode}: high-severity recall {target.high['recall']:.1%}")
    return 0


def cmd_serve(a) -> int:  # pragma: no cover - runs a server
    import uvicorn

    from .api import app_from_env

    uvicorn.run(app_from_env(), host=a.host, port=a.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="contractreview", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="write the synthetic corpus and ground truth")
    g.add_argument("--out", default="data")
    an = sub.add_parser("analyze", help="analyze one .txt or .docx contract")
    an.add_argument("file")
    an.add_argument("--json", action="store_true")
    ev = sub.add_parser("eval", help="evaluate against the ground truth")
    ev.add_argument("--mode", default="baseline", choices=["baseline", "simulated-model", "claude"])
    ev.add_argument("--compare", action="store_true", help="baseline and simulated-model side by side")
    ev.add_argument("--gate", action="store_true")
    ev.add_argument("--min-high-recall", type=float, default=0.8)
    ev.add_argument("--min-recall", type=float, default=0.0)
    ev.add_argument("--out")
    ev.add_argument("--summary")
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8828)
    a = ap.parse_args(argv)
    return {"gen": cmd_gen, "analyze": cmd_analyze, "eval": cmd_eval, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
