"""Section splitting — notebook Section 7.

One MarkdownHeaderTextSplitter for both DOCX and PDF, because markdown.py made
them the same shape. The heading chain each section sits under is recorded as
`section_path`, and that is what eventually becomes the citation you see under
an answer — so anything dropped here is a citation you can never recover later.
"""

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.config import CONFIG, Config
from src.models import StructuralUnit


_PAGE_MARKER_RE = re.compile(r"<!-- PAGE (\d+) -->")


def _get_splitter(config: Config = CONFIG) -> MarkdownHeaderTextSplitter:
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=list(config.markdown_headers),
        strip_headers=False,
    )


def extract_and_strip_page_markers(text: str) -> tuple[str, int | None, int | None]:
    """Pulls the `<!-- PAGE N -->` markers a section inherited from PDF markdown
    conversion back out of the body text, returning (clean_text, page_start, page_end).
    A section can legitimately span more than one page, hence a range rather than
    a single number. No-op for DOCX, which never emits these markers, so
    page_start/page_end both come back None."""
    pages = [int(p) for p in _PAGE_MARKER_RE.findall(text)]
    clean = _PAGE_MARKER_RE.sub("", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    page_start = pages[0] if pages else None
    page_end = pages[-1] if pages else None
    return clean, page_start, page_end


def markdown_to_structural_units(markdown_text: str, document: str, source_type: str,
                                 config: Config = CONFIG) -> list[StructuralUnit]:
    sections = _get_splitter(config).split_text(markdown_text)

    units = []
    for section in sections:
        # section.metadata is keyed by the header *name* ("h1", "h2", ...), not
        # the markdown prefix ("#", "##", ...) — easy to mix up since CONFIG
        # stores (prefix, name) pairs.
        header_keys = [name for _, name in config.markdown_headers]
        present_headers = [(key, section.metadata[key]) for key in header_keys if key in section.metadata]

        title = present_headers[-1][1] if present_headers else None
        level = len(present_headers) if present_headers else None

        clean_text, page_start, page_end = extract_and_strip_page_markers(section.page_content)

        units.append(
            StructuralUnit(
                document=document,
                source_type=source_type,
                title=title,
                level=level,
                text=clean_text,
                metadata={
                    "section_path": [h for _, h in present_headers],
                    "page_start": page_start,
                    "page_end": page_end,
                },
            )
        )

    return units


def build_structural_units(markdown_by_type: dict[str, dict[str, str]],
                           config: Config = CONFIG) -> list[StructuralUnit]:
    """Flatten build_markdown()'s output into one list of sections."""
    units: list[StructuralUnit] = []
    for source_type, by_document in markdown_by_type.items():
        for document, markdown_text in by_document.items():
            units.extend(markdown_to_structural_units(markdown_text, document, source_type, config))
    return units
