"""The provenance contract, tested end to end.

This is the bug that shipped: section metadata was computed correctly, then
dropped at upload time, so every citation read "[Unknown section, p.?]".
These tests assert the round trip — chunk metadata in, citation out — for each
source type, so the two halves can never silently disagree again.
"""

from src.payload import build_payload, content_hash, format_source


def pdf_chunk(**overrides):
    chunk = {
        "document": "Foundation-LLMs.pdf",
        "chunk_id": "unit_111::chunk_24",
        "source_type": "pdf",
        "text": "Training a model to generalize is a fundamental goal.",
        "metadata": {
            "section_path": ["Foundation Models", "Pre-training", "Generalization"],
            "section_title": "Generalization",
            "section_level": 3,
            "page_start": 41,
            "page_end": 42,
            "chunk_index": 24,
        },
    }
    chunk.update(overrides)
    return chunk


class TestBuildPayload:
    def test_flattens_the_heading_chain(self):
        payload = build_payload(pdf_chunk())
        assert payload["section_heading"] == "Foundation Models > Pre-training > Generalization"

    def test_page_range_when_a_section_spans_pages(self):
        assert build_payload(pdf_chunk())["page_label"] == "pp.41–42"

    def test_single_page_label(self):
        chunk = pdf_chunk()
        chunk["metadata"]["page_end"] = 41
        assert build_payload(chunk)["page_label"] == "p.41"

    def test_docx_has_headings_but_no_pages(self):
        chunk = pdf_chunk(document="notes.docx", source_type="docx")
        chunk["metadata"] = {"section_path": ["Setup", "Environment"], "page_start": None, "page_end": None}
        payload = build_payload(chunk)
        assert payload["section_heading"] == "Setup > Environment"
        assert payload["page_label"] == ""

    def test_xlsx_falls_back_to_the_sheet_name(self):
        chunk = pdf_chunk(document="metrics.xlsx", source_type="xlsx", chunk_id="row_7")
        chunk["metadata"] = {"sheet": "Q3 Results", "row": 7}
        assert build_payload(chunk)["section_heading"] == "Q3 Results"

    def test_a_chunk_with_no_metadata_at_all_still_builds(self):
        chunk = pdf_chunk()
        chunk["metadata"] = {}
        payload = build_payload(chunk)
        assert payload["section_heading"] == ""
        assert payload["page_label"] == ""

    def test_extra_metadata_is_carried_through_not_dropped(self):
        chunk = pdf_chunk()
        chunk["metadata"]["custom_field"] = "keep me"
        assert build_payload(chunk)["custom_field"] == "keep me"

    def test_identity_and_content_fields_are_always_present(self):
        payload = build_payload(pdf_chunk())
        for key in ("document", "chunk_id", "source_type", "text", "content_hash",
                    "section_path", "section_heading", "page_label"):
            assert key in payload, f"{key} missing — it would be invisible at query time"

    def test_metadata_cannot_overwrite_an_identity_field(self):
        # setdefault, not update — a stray metadata key must not clobber the payload
        chunk = pdf_chunk()
        chunk["metadata"]["document"] = "WRONG.pdf"
        assert build_payload(chunk)["document"] == "Foundation-LLMs.pdf"


class TestFormatSource:
    def test_round_trip_from_chunk_to_citation(self):
        citation = format_source(build_payload(pdf_chunk()))
        assert citation == "[Foundation-LLMs.pdf — Foundation Models > Pre-training > Generalization, pp.41–42]"

    def test_no_trailing_comma_when_there_is_no_page(self):
        chunk = pdf_chunk(document="notes.docx")
        chunk["metadata"] = {"section_path": ["Setup"], "page_start": None}
        assert format_source(build_payload(chunk)) == "[notes.docx — Setup]"

    def test_unknown_section_only_when_there_genuinely_is_none(self):
        chunk = pdf_chunk()
        chunk["metadata"] = {}
        assert "Unknown section" in format_source(build_payload(chunk))

    def test_chunk_id_is_opt_in(self):
        payload = build_payload(pdf_chunk())
        assert "unit_111::chunk_24" not in format_source(payload)
        assert "unit_111::chunk_24" in format_source(payload, include_chunk_id=True)

    def test_survives_a_raw_payload_missing_everything(self):
        assert format_source({}) == "[Unknown document — Unknown section]"


class TestContentHash:
    def test_is_stable_and_whitespace_insensitive(self):
        assert content_hash("hello") == content_hash("  hello  ")

    def test_differs_when_text_differs(self):
        assert content_hash("hello") != content_hash("hello!")
