"""Read .docx contracts with python-docx. Heading-styled paragraphs become '# Heading' lines for the segmenter."""

from __future__ import annotations

import io


class DocxError(ValueError):
    pass


def docx_to_text(data: bytes) -> str:
    try:
        from docx import Document

        doc = Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises several types for a corrupt package
        raise DocxError(f"not a readable .docx file: {exc}") from exc
    lines: list[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name if p.style is not None else "") or ""
        if style.lower() == "title" and not lines:
            lines += [text, ""]
        elif style.lower().startswith("heading"):
            lines += ["", f"# {text}"]
        else:
            lines.append(text)
    for table in doc.tables:  # order forms often keep the commercial terms in a table
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 2 and cells[0] and cells[1]:
                lines.append(f"{cells[0].rstrip(':')}: {cells[1]}")
    if not lines:
        raise DocxError("the .docx file has no text")
    return "\n".join(lines) + "\n"
