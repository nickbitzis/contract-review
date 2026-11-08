"""Split a contract into sections and clauses with character offsets.

Handles the numbering customer paper actually uses: "1." and "1.1", "1.2." and "1.2)", "Section 4.", "ARTICLE IV -
HEADING", "(a)" subclauses, "§ 7", markdown "# Heading" (what .docx ingestion emits for heading styles), ALL CAPS
headings, inline headings ("9. Limitation of Liability. Except ..."), and "Key: value" lines in order forms.
Numbers are kept as written, so a contract that skips from 7 to 9 still segments correctly.

A clause is the smallest numbered or separated paragraph. Every clause knows the heading of the section it sits
in. Definitions ('"Liability Cap" means ...') are collected so the extractor can resolve defined terms.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

ARTICLE = re.compile(r"^(?:ARTICLE|Article)\s+([IVXLC]+|\d+)\s*(?:[-.:]\s*)?(.*)$")
SECTION = re.compile(r"^(?:Section|SECTION|§)\s*(\d+(?:\.\d+)*)\.?\s+(.*)$")
NUMBER = re.compile(r"^(\d+(?:\.\d+)*)(?:\.\)|[.)])?\s+(.*)$")
ALPHA = re.compile(r"^\(([a-z]{1,4}|\d{1,2})\)\s+(.*)$")
MARKDOWN = re.compile(r"^#{1,6}\s+(.*)$")
KEYVAL = re.compile(r"^([A-Z][A-Za-z/&\- ]{2,40}):\s+(.+)$")
INLINE_HEAD = re.compile(r"^([A-Z][A-Za-z/&,\- ]{1,60}?)\.\s+(\S.*)$")
DEFINITION = re.compile(r'^\s*["“]([^"”]{2,60})["”]\s+(?:means|shall mean|has the meaning)\s+(.*)$')
SMALL = {"of", "and", "for", "the", "to", "in", "on", "or", "a", "an", "by", "with"}


@dataclass
class Clause:
    id: str
    number: str
    heading: str  # section heading this clause belongs to
    text: str
    start: int
    end: int
    label: str = ""  # own short label, for "Key: value" lines
    types: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Segmented:
    title: str
    kind: str
    clauses: list[Clause]
    definitions: dict[str, str]
    sections: list[dict]

    def clause(self, cid: str) -> Clause:
        return next(c for c in self.clauses if c.id == cid)


def _looks_like_heading(s: str) -> bool:
    s = s.strip().rstrip(":")
    if not s or len(s) > 70 or s.endswith((".", ";", ",")):
        return False
    ws = s.split()
    if len(ws) > 8:
        return False
    return all(w[0].isupper() or w.lower() in SMALL or not w[0].isalpha() for w in ws)


def _is_caps_heading(s: str) -> bool:
    s = s.strip()
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters) and len(s.split()) <= 8 and not s.endswith(".")


def _split_inline(rest: str) -> tuple[str, str]:
    """'Limitation of Liability. Except for ...' -> ('Limitation of Liability', 'Except for ...')."""
    m = INLINE_HEAD.match(rest)
    if m and _looks_like_heading(m.group(1)):
        return m.group(1), m.group(2)
    return "", rest


def detect_kind(title: str, text: str) -> str:
    t = title.lower()
    if "order form" in t or re.search(r"^order number:", text[:800], re.I | re.M):
        return "order_form"
    if "data processing" in t or re.search(r"\bdpa\b", t):
        return "dpa"
    return "msa"


def nice_title(s: str) -> str:
    """'FEES AND PAYMENT' -> 'Fees and Payment'."""
    ws = s.lower().split()
    return " ".join(w if (w in SMALL and i) else w[:1].upper() + w[1:] for i, w in enumerate(ws))


def segment(text: str) -> Segmented:
    clauses: list[Clause] = []
    sections: list[dict] = []
    defs: dict[str, str] = {}
    heading, sec_num, title = "", "", ""
    cur: Clause | None = None
    pos = 0

    def new_clause(number: str, body: str, body_start: int, label: str = "") -> Clause:
        c = Clause(f"c{len(clauses) + 1}", number, heading, body, body_start, body_start + len(body), label)
        clauses.append(c)
        return c

    def open_section(num: str, head: str, start: int) -> None:
        nonlocal heading, sec_num
        heading, sec_num = head.strip(), num
        sections.append({"number": num, "heading": heading, "start": start})

    for raw in text.split("\n"):
        line_start = pos
        pos += len(raw) + 1
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            cur = None
            continue
        lead = len(line) - len(line.lstrip())
        if not title:
            title = stripped
            continue

        def body_at(body: str, _ls=line_start, _line=line, _lead=lead) -> int:
            return _ls + _line.index(body, _lead) if body else _ls + len(_line)

        m = ARTICLE.match(stripped)
        if m:
            open_section(m.group(1), nice_title(m.group(2).strip(" -")) if m.group(2).isupper() else m.group(2).strip(" -"),
                         line_start)
            cur = None
            continue
        m = MARKDOWN.match(stripped)
        if m:
            open_section("", m.group(1), line_start)
            cur = None
            continue
        m = SECTION.match(stripped) or NUMBER.match(stripped)
        if m:
            num, rest = m.group(1), m.group(2).strip()
            top = "." not in num and not ARTICLE.match(stripped)
            if top and _looks_like_heading(rest):
                open_section(num, rest, line_start)
                cur = None
                continue
            if top:
                head, body = _split_inline(rest)
                if head:
                    open_section(num, head, line_start)
                cur = new_clause(num, body, body_at(body))
            else:
                cur = new_clause(num, rest, body_at(rest))
            continue
        m = ALPHA.match(stripped)
        if m:
            body = m.group(2).strip()
            cur = new_clause(f"{sec_num}({m.group(1)})" if sec_num else f"({m.group(1)})", body, body_at(body))
            continue
        if _is_caps_heading(stripped) and len(stripped) < 60:
            open_section("", nice_title(stripped), line_start)
            cur = None
            continue
        m = KEYVAL.match(stripped)
        if m and len(m.group(1).split()) <= 5 and not DEFINITION.match(stripped):
            cur = new_clause("", stripped, body_at(stripped), label=m.group(1).strip())
            cur = None  # one line per field
            continue
        if cur is not None and not cur.text.rstrip().endswith((".", ";", ":", "!", "?")):
            # continuation line of the current paragraph
            cur.text = text[cur.start: line_start + len(line)]
            cur.end = line_start + len(line)
            continue
        cur = new_clause(sec_num, stripped, body_at(stripped))

    for c in clauses:
        if "definition" in c.heading.lower():
            m = DEFINITION.match(c.text)
            if m:
                defs[m.group(1).strip()] = m.group(2).strip()
    return Segmented(title, detect_kind(title, text), clauses, defs, sections)


def sentences(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """Sentence split that keeps '1.5%', '(a)', and 'Inc.' inside sentences. Offsets are absolute."""
    out = []
    for m in re.finditer(r"(?:[^.!?]|\.(?=\d)|\.(?=[a-z,])|(?<=\bInc)\.|(?<=\be\.g)\.|(?<=\bi\.e)\.)+[.!?]?", text):
        s = m.group(0)
        if not s.strip():
            continue
        lead = len(s) - len(s.lstrip())
        st = s.strip()
        out.append((st, offset + m.start() + lead, offset + m.start() + lead + len(st)))
    return out
