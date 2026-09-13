"""Chunking checks from the notebook's 'Check the outcome: chunking' cell.

The two assertions that cell was really making: no chunk exceeds the max size,
and no chunk loses its section provenance.
"""

import pytest

from src.chunking import (
    chunk_structural_unit, find_chunk_boundary, split_into_sentences, xlsx_records_to_chunks,
)
from src.models import StructuralUnit


def unit(text: str) -> StructuralUnit:
    return StructuralUnit(
        document="doc.pdf",
        source_type="pdf",
        title="Generalization",
        level=3,
        text=text,
        metadata={"section_path": ["Models", "Generalization"], "page_start": 5, "page_end": 5},
    )


class TestSplitIntoSentences:
    def test_splits_on_sentence_ends(self):
        assert len(split_into_sentences("One thing. Two things. Three things.")) == 3

    def test_does_not_split_mid_sentence_on_a_lowercase_start(self):
        # "e.g. this" should not become two sentences
        assert len(split_into_sentences("Models generalize, e.g. across tasks.")) == 1

    def test_empty_text_gives_no_sentences(self):
        assert split_into_sentences("   ") == []


class TestFindChunkBoundary:
    def test_prefers_the_last_sentence_end(self):
        text = "First sentence. Second sentence. Third"
        boundary = find_chunk_boundary(text, 0, 33)
        assert text[:boundary].rstrip().endswith(".")

    def test_falls_back_to_a_word_boundary_on_a_run_on(self):
        # no sentence end anywhere, so it returns the index OF the last space
        text = "word " * 40
        boundary = find_chunk_boundary(text, 0, 100)
        assert text[boundary] == " "
        assert not text[:boundary].endswith("wor")   # didn't cut mid-word

    def test_returns_end_when_there_is_no_break_at_all(self):
        text = "x" * 200
        assert find_chunk_boundary(text, 0, 100) == 100


class TestChunkStructuralUnit:
    def test_short_section_becomes_exactly_one_chunk(self):
        chunks = chunk_structural_unit(unit("Short text."), chunk_size=1500)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_empty_section_produces_nothing(self):
        assert chunk_structural_unit(unit("   ")) == []

    def test_long_section_is_split(self):
        chunks = chunk_structural_unit(unit("This is a sentence. " * 300), chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_the_size_limit_by_more_than_the_boundary_search(self):
        chunks = chunk_structural_unit(unit("This is a sentence. " * 300), chunk_size=500, overlap=50)
        assert all(len(c["text"]) <= 500 for c in chunks)

    def test_every_chunk_keeps_its_section_provenance(self):
        # this is the check that would have caught the "Unknown section" bug
        # one stage earlier than the payload
        chunks = chunk_structural_unit(unit("This is a sentence. " * 300), chunk_size=500)
        for chunk in chunks:
            assert chunk["metadata"]["section_path"] == ["Models", "Generalization"]
            assert chunk["metadata"]["section_title"] == "Generalization"
            assert chunk["metadata"]["page_start"] == 5

    def test_chunk_indexes_are_sequential(self):
        chunks = chunk_structural_unit(unit("This is a sentence. " * 300), chunk_size=500)
        assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_makes_progress_and_terminates_on_text_with_no_spaces(self):
        # a pathological input that could loop forever if start never advances
        chunks = chunk_structural_unit(unit("x" * 5000), chunk_size=500, overlap=50)
        assert len(chunks) > 1


class TestXlsxRecordsToChunks:
    def test_one_chunk_per_row(self):
        records = [
            {"document": "m.xlsx", "text": "a", "metadata": {"sheet": "S", "row": 2}},
            {"document": "m.xlsx", "text": "b", "metadata": {"sheet": "S", "row": 3}},
        ]
        chunks = xlsx_records_to_chunks(records)
        assert len(chunks) == 2
        assert [c.chunk_id for c in chunks] == ["row_0", "row_1"]
        assert all(c.source_type == "xlsx" for c in chunks)

    def test_row_metadata_is_preserved(self):
        records = [{"document": "m.xlsx", "text": "a", "metadata": {"sheet": "Q3", "row": 2}}]
        assert xlsx_records_to_chunks(records)[0].metadata["sheet"] == "Q3"
