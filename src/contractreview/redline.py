"""Render a redlined document: the contract text with tracked-changes style insertions and deletions in HTML."""

from __future__ import annotations

import difflib
import html
import re

CSS = """
:root { --bg:#fff; --text:#1b1f24; --muted:#5c6670; --ins:#1d7a46; --ins-bg:#e3f4ea; --del:#b42318; --del-bg:#fde8e6; }
@media (prefers-color-scheme: dark) { :root { --bg:#111418; --text:#e6e9ed; --muted:#9aa4af; --ins:#5fd08f;
  --ins-bg:#16301f; --del:#ff8a80; --del-bg:#3a1a18; } }
body { background:var(--bg); color:var(--text); font:15px/1.6 Georgia, serif; max-width:820px; margin:0 auto; padding:16px; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; font:inherit; }
ins { color:var(--ins); background:var(--ins-bg); text-decoration:underline; }
del { color:var(--del); background:var(--del-bg); text-decoration:line-through; }
.note { color:var(--muted); font:13px/1.4 system-ui, sans-serif; border-bottom:1px solid var(--muted); padding-bottom:8px; }
sup { color:var(--muted); font:11px system-ui, sans-serif; }
"""


def word_diff(original: str, proposed: str) -> str:
    """Word-level diff as HTML with <del> and <ins>."""
    a, b = re.findall(r"\S+|\s+", original), re.findall(r"\S+|\s+", proposed)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.ratio() < 0.6:  # mostly rewritten: show one deletion and one insertion, easier to read
        return f"<del>{html.escape(original)}</del> <ins>{html.escape(proposed)}</ins>"
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(html.escape("".join(a[i1:i2])))
            continue
        if i2 > i1:
            out.append(f"<del>{html.escape(''.join(a[i1:i2]))}</del>")
        if j2 > j1:
            out.append(f"<ins>{html.escape(''.join(b[j1:j2]))}</ins>")
    return "".join(out)


def apply_changes(text: str, changes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep non-overlapping changes in document order. Returns (applied, skipped)."""
    applied, skipped, last_end = [], [], -1
    for ch in sorted(changes, key=lambda c: (c["start"], c["end"])):
        if ch["start"] < last_end:
            skipped.append(ch)
            continue
        applied.append(ch)
        last_end = max(last_end, ch["end"])
    return applied, skipped


def final_text(text: str, changes: list[dict]) -> str:
    applied, _ = apply_changes(text, changes)
    out, pos = [], 0
    for ch in applied:
        out.append(text[pos:ch["start"]])
        out.append(ch["text"])
        pos = ch["end"]
    out.append(text[pos:])
    return "".join(out)


def redlined_html(title: str, text: str, changes: list[dict], note: str = "") -> str:
    """changes: [{start, end, text, issue_id}] where text replaces text[start:end] (start == end inserts)."""
    applied, skipped = apply_changes(text, changes)
    parts, pos = [], 0
    for ch in applied:
        parts.append(html.escape(text[pos:ch["start"]]))
        original = text[ch["start"]:ch["end"]]
        if original:
            parts.append(word_diff(original, ch["text"]))
        else:
            parts.append(f"<ins>{html.escape(ch['text'])}</ins>")
        parts.append(f"<sup>[{html.escape(ch.get('issue_id', ''))}]</sup>")
        pos = ch["end"]
    parts.append(html.escape(text[pos:]))
    extra = f" {len(skipped)} overlapping change(s) not shown." if skipped else ""
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)} (redline)</title><style>{CSS}</style></head><body>"
            f"<p class=\"note\">Redline of {html.escape(title)}. {len(applied)} change(s). Insertions underlined, "
            f"deletions struck through.{extra} {html.escape(note)}</p><pre>{''.join(parts)}</pre></body></html>")
