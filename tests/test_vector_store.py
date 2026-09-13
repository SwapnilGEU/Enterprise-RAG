"""Deduplication logic, tested without a live Qdrant.

classify_chunks is pure apart from one call to fetch_all_existing_state, so we
monkeypatch that and assert the four buckets directly.
"""

import pytest

from src import vector_store
from src.payload import content_hash


EXISTING = {
    "a.pdf": {
        # stored without section metadata — written before build_payload existed
        "u0::c0": {"hash": content_hash("text zero"), "has_section_metadata": False},
        # fully current
        "u0::c1": {"hash": content_hash("text one"), "has_section_metadata": True},
        # text has since changed
        "u0::c2": {"hash": content_hash("OLD text two"), "has_section_metadata": True},
    }
}


@pytest.fixture
def fake_collection(monkeypatch):
    monkeypatch.setattr(vector_store, "fetch_all_existing_state", lambda config=None: EXISTING)


def chunk(document, chunk_id, text):
    return {"document": document, "chunk_id": chunk_id, "source_type": "pdf",
            "text": text, "metadata": {}}


class TestClassifyChunks:
    def test_sorts_every_chunk_into_the_right_bucket(self, fake_collection):
        buckets = vector_store.classify_chunks([
            chunk("a.pdf", "u0::c0", "text zero"),      # stale metadata
            chunk("a.pdf", "u0::c1", "text one"),       # unchanged
            chunk("a.pdf", "u0::c2", "NEW text two"),   # changed
            chunk("b.docx", "u1::c0", "brand new"),     # new (unseen document)
            chunk("a.pdf", "u9::c9", "unseen chunk"),   # new (unseen chunk id)
        ])
        assert [c["chunk_id"] for c in buckets["stale_metadata"]] == ["u0::c0"]
        assert [c["chunk_id"] for c in buckets["unchanged"]] == ["u0::c1"]
        assert [c["chunk_id"] for c in buckets["changed"]] == ["u0::c2"]
        assert {c["chunk_id"] for c in buckets["new"]} == {"u1::c0", "u9::c9"}

    def test_stale_metadata_is_queued_for_re_upload(self, fake_collection):
        """The whole point of the fourth bucket: a pure hash check would call
        these 'unchanged' forever and they would never gain their headings."""
        buckets = vector_store.classify_chunks([chunk("a.pdf", "u0::c0", "text zero")])
        to_embed = buckets["new"] + buckets["changed"] + buckets["stale_metadata"]
        assert len(to_embed) == 1

    def test_empty_input_gives_four_empty_buckets(self, fake_collection):
        buckets = vector_store.classify_chunks([])
        assert set(buckets) == {"new", "changed", "stale_metadata", "unchanged"}
        assert all(v == [] for v in buckets.values())


class TestPointId:
    def test_is_deterministic(self):
        first = vector_store.point_id("a.pdf", "u0::c0")
        second = vector_store.point_id("a.pdf", "u0::c0")
        assert first == second

    def test_differs_per_document_and_per_chunk(self):
        assert vector_store.point_id("a.pdf", "u0::c0") != vector_store.point_id("b.pdf", "u0::c0")
        assert vector_store.point_id("a.pdf", "u0::c0") != vector_store.point_id("a.pdf", "u0::c1")
