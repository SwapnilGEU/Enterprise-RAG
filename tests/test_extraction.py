"""Checks from the notebook's 'Check the outcome: ingestion & cleaning' cell,
turned into assertions. All pure functions — no files, no network."""

from src.extraction import (
    classify_xlsx_columns, clean_text, compute_stats, find_boilerplate_lines, make_record,
)


class TestCleanText:
    def test_normalises_line_endings(self):
        assert clean_text("a\r\nb\rc") == "a\nb\nc"

    def test_collapses_runs_of_spaces_and_tabs(self):
        assert clean_text("a   \t  b") == "a b"

    def test_joins_a_markdown_hard_break_rather_than_keeping_it(self):
        # trailing spaces before a newline are a markdown hard-break; the
        # pipeline treats them as an artifact of PDF line wrapping, not a break
        assert clean_text("word   \nnext") == "word\nnext"

    def test_collapses_three_or_more_newlines_to_a_paragraph_break(self):
        assert clean_text("a\n\n\n\nb") == "a\n\nb"

    def test_strips_surrounding_whitespace(self):
        assert clean_text("  hello  ") == "hello"


class TestMakeRecord:
    def test_record_has_the_fields_every_downstream_stage_reads(self):
        record = make_record("f.pdf", "pdf", "page_1_line_2", " some text ", {"page": 1})
        assert record["document"] == "f.pdf"
        assert record["source_type"] == "pdf"
        assert record["location"] == "page_1_line_2"
        assert record["text"] == "some text"
        assert record["metadata"] == {"page": 1}

    def test_stats_are_computed_on_the_cleaned_text(self):
        record = make_record("f.pdf", "pdf", "loc", "  two words  ")
        assert record["char_count"] == len("two words")
        assert record["word_count"] == 2

    def test_metadata_defaults_to_empty_dict_not_none(self):
        assert make_record("f.pdf", "pdf", "loc", "x")["metadata"] == {}


def test_compute_stats_counts_lines():
    assert compute_stats("a\nb\nc")["line_count"] == 3


class TestFindBoilerplateLines:
    def test_too_few_pages_to_judge_returns_nothing(self):
        assert find_boilerplate_lines(["header\nbody", "header\nbody"]) == set()

    def test_detects_a_running_header_repeated_across_pages(self):
        pages = ["Chapter One\nreal content here"] * 5
        assert "Chapter One" in find_boilerplate_lines(pages)

    def test_page_numbers_are_normalised_before_comparing(self):
        # the number changes every page but the header text doesn't — exact
        # string matching alone would never catch this
        pages = [f"{n} Prompting\nbody text for page {n}" for n in range(1, 8)]
        boilerplate = find_boilerplate_lines(pages)
        assert "1 Prompting" in boilerplate
        assert "7 Prompting" in boilerplate

    def test_real_content_is_not_stripped(self):
        pages = ["Header\nunique body one", "Header\nunique body two", "Header\nunique body three"]
        boilerplate = find_boilerplate_lines(pages)
        assert "unique body one" not in boilerplate


class TestClassifyXlsxColumns:
    def test_long_free_text_column_is_semantic(self):
        headers = ["notes"]
        rows = [("this is a reasonably long sentence of free text",)] * 5
        assert classify_xlsx_columns(headers, rows)["notes"] == "semantic"

    def test_short_values_are_metadata(self):
        headers = ["status"]
        rows = [("open",), ("closed",), ("open",)]
        assert classify_xlsx_columns(headers, rows)["status"] == "metadata"

    def test_id_like_name_is_metadata_even_with_long_values(self):
        headers = ["id"]
        rows = [("a long descriptive value that would otherwise look semantic",)] * 5
        assert classify_xlsx_columns(headers, rows)["id"] == "metadata"

    def test_empty_column_does_not_crash(self):
        assert classify_xlsx_columns(["blank"], [(None,), (None,)])["blank"] == "metadata"
