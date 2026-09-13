"""Markdown conversion — notebook Section 6.

DOCX and PDF both become a single Markdown string per document, using whatever
heading signal the format actually has: `style` for DOCX, `font_size`/`is_bold`
for PDF, with the same numbered-heading fallback for both. Producing Markdown
from both formats is what lets one splitter handle both in sections.py.
"""

import re
from collections import Counter

from src.config import CONFIG, Config


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------

def detect_docx_structure(record: dict) -> int | None:
    """Heading level for a DOCX record, or None if it's body text."""
    style = record.get("metadata", {}).get("style", "")

    if style == "Heading 1":
        return 1
    if style == "Heading 2":
        return 2

    text = record.get("text", "").strip()
    match = re.match(r"^(\d+(?:\.\d+)*)\s+.+$", text)
    if match:
        return match.group(1).count(".") + 1

    return None


def docx_records_to_markdown(records: list[dict]) -> str:
    lines = []
    for record in records:
        level = detect_docx_structure(record)
        text = record["text"]
        if level is not None:
            lines.append(f"{'#' * min(level, 6)} {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def compute_body_font_size(lines: list[dict]) -> float:
    """Most common font size in the document — the body-text baseline headings
    are measured against."""
    if not lines:
        return 0
    return Counter(l["metadata"]["font_size"] for l in lines).most_common(1)[0][0]


_NON_HEADING_PREFIX_RE = re.compile(r"^(FIGURE|TABLE)\b", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,4})\s+\S")


def is_probable_heading_text(text: str, max_words: int = 12) -> bool:
    """Extra shape check so 'big font + bold' alone can't promote a stray caption,
    pull-quote, or run-on sentence to a heading. A numbered heading (e.g. "2.1 Foo")
    is exempt from the length/punctuation checks since the number itself is a
    strong signal."""
    text = text.strip()
    if not text or _NON_HEADING_PREFIX_RE.match(text):
        return False

    if _NUMBERED_HEADING_RE.match(text):
        return True

    if text.endswith((".", ";", ":")):
        return False
    if len(text.split()) > max_words:
        return False

    return True


def detect_pdf_heading_level(record: dict, body_font_size: float,
                             config: Config = CONFIG) -> int | None:
    """Heading level for a PDF line, or None if it's body text.

    Primary signal: font size relative to the body-text baseline (bigger => higher-level
    heading — the measured equivalent of DOCX's Heading 1/2 styles).
    Fallback signal: a numbered heading, same fallback used for DOCX.
    Both signals also have to pass is_probable_heading_text — size/boldness alone
    isn't enough, since bold captions, callouts, and pull-quotes are common false
    positives at the same font size as a real heading. A lone page-number footer
    (e.g. "154") would also pass the numbered-heading regex on its own — that's
    handled upstream in find_boilerplate_lines instead, since by the time a record
    reaches here we can't tell a page number from a real "1 Introduction" by shape.
    """
    text = record.get("text", "").strip()
    if not is_probable_heading_text(text):
        return None

    font_size = record["metadata"]["font_size"]
    is_bold = record["metadata"]["is_bold"]
    size_ratio = font_size / body_font_size if body_font_size else 1.0

    if size_ratio >= config.pdf_heading_size_ratio_h1:
        return 1
    if size_ratio >= config.pdf_heading_size_ratio_h2:
        return 2

    match = _NUMBERED_HEADING_RE.match(text)
    if match and (is_bold or size_ratio >= 1.0):
        return match.group(1).count(".") + 1

    return None


def pdf_records_to_markdown(records: list[dict], config: Config = CONFIG) -> str:
    """Converts PDF line-records to Markdown.

    Each record is one *visual* line as PyMuPDF laid it out — most are a paragraph
    wrapping at the page width, not a real paragraph break. So consecutive body
    lines are buffered and joined with a single space; a paragraph only breaks at
    a heading or a page change. A `<!-- PAGE N -->` marker goes in on every page
    change — sections.py reads it back out to attach page numbers to metadata.
    """
    body_font_size = compute_body_font_size(records)

    lines_out: list[str] = []
    paragraph_buf: list[str] = []
    current_page = None

    def flush_paragraph():
        if paragraph_buf:
            lines_out.append(" ".join(paragraph_buf))
            paragraph_buf.clear()

    for record in records:
        page = record["metadata"]["page"]
        if page != current_page:
            flush_paragraph()
            lines_out.append(f"<!-- PAGE {page} -->")
            current_page = page

        level = detect_pdf_heading_level(record, body_font_size, config)
        text = record["text"]

        if level is not None:
            flush_paragraph()
            lines_out.append(f"{'#' * min(level, 6)} {text}")
        else:
            paragraph_buf.append(text)

    flush_paragraph()
    return "\n\n".join(lines_out)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_markdown(documents: list[dict], config: Config = CONFIG) -> dict[str, dict[str, str]]:
    """Group flat records by source type and document, then convert each to
    Markdown. Returns {"docx": {filename: markdown}, "pdf": {filename: markdown}}.
    XLSX is absent on purpose — a row is already an atomic unit and never goes
    through heading detection or section splitting.
    """
    result: dict[str, dict[str, str]] = {"docx": {}, "pdf": {}}

    converters = {"docx": docx_records_to_markdown, "pdf": lambda r: pdf_records_to_markdown(r, config)}

    for source_type, convert in converters.items():
        records = [d for d in documents if d["source_type"] == source_type]
        for name in sorted({d["document"] for d in records}):
            result[source_type][name] = convert([d for d in records if d["document"] == name])

    return result
