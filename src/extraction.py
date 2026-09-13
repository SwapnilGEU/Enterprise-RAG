"""Document extraction — notebook Sections 3, 4 and 5.

One record per natural unit of each format: a line for PDF, a paragraph for
DOCX, a row for XLSX. Line-level PDF records are what heading detection in
markdown.py needs (font size and boldness live per line, not per page).
"""

import json
import re
from collections import Counter
from pathlib import Path

from docx import Document as DocxDocument   # aliased: LangChain also has a Document
from openpyxl import load_workbook
import pymupdf

from src.config import CONFIG, Config, logger


# --------------------------------------------------------------------------
# Shared helpers (Section 4.1)
# --------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing spaces before a newline (markdown hard-break) — join, don't preserve as a break
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compute_stats(text: str) -> dict:
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(text.splitlines()),
    }


def make_record(document: str, source_type: str, location: str, text: str,
                metadata: dict | None = None) -> dict:
    cleaned = clean_text(text)
    record = {
        "document": document,
        "source_type": source_type,
        "location": location,
        "text": cleaned,
        "metadata": metadata or {},
    }
    record.update(compute_stats(cleaned))
    return record


# --------------------------------------------------------------------------
# PDF (Section 4.2)
# --------------------------------------------------------------------------

_LEADING_OR_TRAILING_NUMBER_RE = re.compile(r"^\d+\s+|\s+\d+$")


def find_boilerplate_lines(page_texts: list[str], config: Config = CONFIG) -> set[str]:
    """Lines repeating across a large fraction of a document's pages — running
    headers/footers rather than content. Normalizes a leading/trailing page
    number (e.g. "154   Prompting" / "Prompting   154") before comparing, since
    the number changes every page but the surrounding header/footer text doesn't —
    exact-string matching alone would never catch it, and it would silently leak
    into chunk text as body content."""
    if len(page_texts) < 3:
        return set()

    def normalize(line: str) -> str:
        return _LEADING_OR_TRAILING_NUMBER_RE.sub("", line).strip()

    line_counts = Counter()
    original_by_normalized: dict[str, set[str]] = {}
    for text in page_texts:
        unique_lines_on_page = {line.strip() for line in text.splitlines() if line.strip()}
        for line in unique_lines_on_page:
            key = normalize(line)
            if key:
                line_counts[key] += 1
                original_by_normalized.setdefault(key, set()).add(line)

    threshold = max(2, int(len(page_texts) * config.pdf_boilerplate_min_repeat_ratio))
    boilerplate: set[str] = set()
    for key, count in line_counts.items():
        if count >= threshold:
            # every page's numbered variant: "153 Prompting", "154 Prompting", ...
            boilerplate.update(original_by_normalized[key])
    return boilerplate


def extract_pdf(file_path: Path, config: Config = CONFIG) -> list[dict]:
    with pymupdf.open(file_path) as doc:
        page_texts = [page.get_text() for page in doc]
        boilerplate = find_boilerplate_lines(page_texts, config)
        if boilerplate:
            logger.info(f"{file_path.name}: stripping {len(boilerplate)} repeated header/footer line(s)")

        documents = []
        for page_number, page in enumerate(doc, start=1):
            for line_number, block in enumerate(page.get_text("dict")["blocks"], start=1):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    raw_text = "".join(span["text"] for span in spans).strip()
                    if not raw_text or raw_text in boilerplate:
                        continue

                    documents.append(
                        make_record(
                            document=file_path.name,
                            source_type="pdf",
                            location=f"page_{page_number}_line_{line_number}",
                            text=raw_text,
                            metadata={
                                "page": page_number,
                                "font_size": round(max(s["size"] for s in spans), 1),
                                "is_bold": any("bold" in s["font"].lower() for s in spans),
                            },
                        )
                    )

    return documents


# --------------------------------------------------------------------------
# DOCX (Section 4.3)
# --------------------------------------------------------------------------

def extract_docx(file_path: Path) -> list[dict]:
    documents = []
    doc = DocxDocument(file_path)

    for index, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            documents.append(
                make_record(
                    document=file_path.name,
                    source_type="docx",
                    location=f"paragraph_{index}",
                    text=text,
                    metadata={"style": paragraph.style.name},
                )
            )

    return documents


# --------------------------------------------------------------------------
# XLSX (Section 4.4)
# --------------------------------------------------------------------------

def classify_xlsx_columns(headers: list[str], data_rows: list[tuple],
                          config: Config = CONFIG) -> dict[str, str]:
    """Decide per column whether its values are free text ("semantic", joined
    into the chunk text) or identifiers/categories ("metadata", attached as
    structured fields instead of being mashed into the text)."""
    columns = {header: [] for header in headers}
    for row in data_rows:
        for header, value in zip(headers, row):
            columns[header].append(value)

    classification = {}
    for header, values in columns.items():
        non_empty = [str(v).strip() for v in values if v is not None and str(v).strip()]
        avg_word_count = (
            sum(len(v.split()) for v in non_empty) / len(non_empty)
            if non_empty else 0
        )

        looks_like_id = header.strip().lower() in config.xlsx_id_like_names
        looks_short = avg_word_count < config.xlsx_semantic_min_avg_words

        classification[header] = "metadata" if (looks_like_id or looks_short) else "semantic"

    return classification


def read_xlsx_rows(file_path: Path) -> list[tuple[str, list[str], list[tuple]]]:
    workbook = load_workbook(file_path, read_only=True, data_only=True)
    sheets = []

    for sheet in workbook.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue

        headers = [
            str(value).strip() if value is not None else f"column_{i}"
            for i, value in enumerate(rows[0])
        ]
        sheets.append((sheet.title, headers, rows[1:]))

    workbook.close()
    return sheets


def extract_xlsx(file_path: Path, config: Config = CONFIG) -> list[dict]:
    documents = []

    for sheet_title, headers, data_rows in read_xlsx_rows(file_path):
        column_roles = classify_xlsx_columns(headers, data_rows, config)
        semantic_headers = [h for h in headers if column_roles[h] == "semantic"]
        metadata_headers = [h for h in headers if column_roles[h] == "metadata"]

        logger.info(f"{file_path.name} [{sheet_title}]: semantic={semantic_headers} metadata={metadata_headers}")

        for row_number, row in enumerate(data_rows, start=2):
            row_dict = dict(zip(headers, row))

            semantic_values = [
                str(row_dict[h]).strip()
                for h in semantic_headers
                if row_dict.get(h) is not None and str(row_dict[h]).strip()
            ]
            if not semantic_values:
                continue

            text = " | ".join(semantic_values)

            row_metadata = {"sheet": sheet_title, "row": row_number}
            for h in metadata_headers:
                if row_dict.get(h) is not None:
                    row_metadata[h] = row_dict[h]

            documents.append(
                make_record(
                    document=file_path.name,
                    source_type="xlsx",
                    location=f"{sheet_title}!row_{row_number}",
                    text=text,
                    metadata=row_metadata,
                )
            )

    return documents


# --------------------------------------------------------------------------
# Inventory + orchestration (Sections 3 and 5)
# --------------------------------------------------------------------------

EXTRACTORS = {".pdf": extract_pdf, ".docx": extract_docx, ".xlsx": extract_xlsx}


def ingest_file(file_path: Path) -> list[dict]:
    extractor = EXTRACTORS.get(file_path.suffix.lower())
    if extractor is None:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    return extractor(file_path)


def list_data_files(config: Config = CONFIG) -> list[Path]:
    config.ensure_dirs()
    files = sorted(config.raw_dir.iterdir())

    logger.info(f"Files in raw directory: {len(files)}")
    for file in files:
        size_kb = file.stat().st_size / 1024
        logger.info(f"- {file.name} | {file.suffix or 'no ext'} | {size_kb:.2f} KB")

    return files


def run_ingestion(config: Config = CONFIG) -> list[dict]:
    all_documents = []

    for file_path in sorted(config.raw_dir.iterdir()):
        if file_path.suffix.lower() not in config.supported_extensions:
            continue
        try:
            extracted = ingest_file(file_path)
            all_documents.extend(extracted)
            logger.info(f"{file_path.name}: {len(extracted)} records")
        except Exception:
            logger.exception(f"Failed to ingest {file_path.name}, skipping.")

    logger.info(f"Total records: {len(all_documents)}")
    return all_documents


def save_documents(documents: list[dict], config: Config = CONFIG,
                   filename: str = "ingested_records.json") -> Path:
    config.ensure_dirs()
    out_path = config.processed_dir / filename
    out_path.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved {len(documents)} records to {out_path}")
    return out_path


def load_documents(config: Config = CONFIG, filename: str = "ingested_records.json") -> list[dict]:
    """Reload records from disk instead of re-running extraction — for picking
    work back up without re-parsing every file."""
    path = config.processed_dir / filename
    return json.loads(path.read_text(encoding="utf-8"))
