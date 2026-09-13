"""Shared dataclasses.

These live in their own module purely to keep imports one-directional.
`Chunk` is needed by chunking.py, payload.py and vector_store.py; if it lived
in chunking.py then vector_store would import chunking, and the first time
chunking needed anything from the store you would have an import cycle Python
cannot resolve. This module imports nothing from the project.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StructuralUnit:
    """One section of a document, as cut by the Markdown heading splitter."""
    document: str
    source_type: str
    title: str | None
    level: int | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """One embeddable piece of text, with the provenance it inherited from its
    section. `metadata` is what payload.build_payload turns into a Qdrant payload."""
    chunk_id: str
    document: str
    source_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "source_type": self.source_type,
            "text": self.text,
            "metadata": self.metadata,
        }
