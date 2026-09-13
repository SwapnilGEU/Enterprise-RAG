"""Configuration and logging — notebook Section 2.

One Config object, one logger, both imported by every other module.
Secrets come from the environment (see .env.example); nothing secret is
hardcoded here, so this file is safe to commit.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = the folder containing src/. Anchoring to __file__ rather than
# Path.cwd() means scripts work no matter which directory you run them from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    base_dir: Path = PROJECT_ROOT
    data_subdir: str = "data"
    raw_subdir: str = "raw"                # data/raw       — untouched source files
    processed_subdir: str = "processed"    # data/processed — ingestion + chunking output
    output_subdir: str = "outputs"
    log_subdir: str = "logs"

    supported_extensions: tuple = (".pdf", ".docx", ".xlsx")

    short_page_word_threshold: int = 20

    # A line that repeats across at least this fraction of a PDF's pages is
    # treated as a running header/footer and stripped.
    pdf_boilerplate_min_repeat_ratio: float = 0.4

    # PDF heading detection: a line's font_size divided by the document's body
    # font size, compared against these ratios.
    pdf_heading_size_ratio_h1: float = 1.4
    pdf_heading_size_ratio_h2: float = 1.15

    # An xlsx column whose non-empty values average fewer words than this is
    # treated as an identifier/category rather than free text.
    xlsx_semantic_min_avg_words: float = 3.0
    xlsx_id_like_names: frozenset = frozenset({
        "id", "row_id", "index", "rank", "ranking", "serial", "sr_no", "s_no",
    })

    # Markdown heading levels the section splitter recognizes (#, ##, ###, ####).
    markdown_headers: tuple = (("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"))

    # ---- Chunking -------------------------------------------------------
    embedding_model_name: str = "all-MiniLM-L6-v2"   # local model, boundary detection only
    semantic_chunk_percentile: float = 95.0     # higher = fewer, larger chunks
    semantic_chunk_min_sentences: int = 4       # below this, keep the section as one chunk
    semantic_chunk_max_chars: int = 1500        # hard ceiling; oversized chunks get split
    fixed_chunk_overlap: int = 200

    # ---- Qdrant ---------------------------------------------------------
    # The URL is not secret. The API key is environment-only: if it is missing,
    # get_client() raises with instructions rather than failing obscurely.
    qdrant_url: str = field(default_factory=lambda: os.environ.get(
        "QDRANT_URL",
        "https://bd42146b-72c4-4a22-a4db-84c41cd50634.us-east-2-0.aws.cloud.qdrant.io",
    ))
    qdrant_api_key: str = field(default_factory=lambda: os.environ.get("QDRANT_API_KEY", ""))
    collection_name: str = field(default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "RAG-hybrid-search"))

    # Qdrant Cloud computes these server-side (cloud_inference=True), so nothing
    # needs fastembed or a GPU locally.
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "qdrant/bm25"
    late_interaction_model: str = "answerdotai/answerai-colbert-small-v1"  # ColBERT — reranking, not ANN

    dense_vector_size: int = 384
    late_interaction_vector_size: int = 96

    hybrid_prefetch_limit: int = 20    # candidates per branch (dense, sparse) before fusion
    final_top_k: int = 5
    dedup_namespace: str = "12345678-1234-5678-1234-567812345678"  # fixed once, never change

    # ---- Ollama ---------------------------------------------------------
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct"))
    ollama_base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_num_predict: int = 512
    # Ollama defaults num_ctx to 4096 regardless of what the model supports, and
    # num_predict comes OUT of that budget. 5 retrieved chunks plus an agent loop
    # overflows 4096 easily. Lower this to 8192 if you run out of VRAM.
    ollama_num_ctx: int = 16384
    ollama_temperature: float = 0.0

    # ---- Retry & graceful failure --------------------------------------
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 0.5
    retry_backoff_factor: float = 2.0     # exponential: 0.5s, 1s, 2s, ...
    retry_max_delay_seconds: float = 8.0
    retry_jitter_seconds: float = 0.25    # avoid thundering-herd on shared services

    # ---- Postgres -------------------------------------------------------
    pg_host: str = field(default_factory=lambda: os.environ.get("PG_HOST", "localhost"))
    pg_port: str = field(default_factory=lambda: os.environ.get("PG_PORT", "5432"))
    pg_dbname: str = field(default_factory=lambda: os.environ.get("PG_DBNAME", "RAG"))
    pg_user: str = field(default_factory=lambda: os.environ.get("PG_USER", "postgres"))
    pg_password: str = field(default_factory=lambda: os.environ.get("PG_PASSWORD", "admin"))
    pg_connect_timeout_seconds: int = 5

    # ---- Derived paths --------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.data_subdir

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / self.raw_subdir

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / self.processed_subdir

    @property
    def output_dir(self) -> Path:
        return self.base_dir / self.output_subdir

    @property
    def log_dir(self) -> Path:
        return self.base_dir / self.log_subdir

    @property
    def pg_uri(self) -> str:
        """SQLAlchemy URI for the SQL agent — built from the same settings
        history.py connects with, so the two can never drift apart."""
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_dbname}"
        )

    def ensure_dirs(self) -> None:
        for directory in (self.raw_dir, self.processed_dir, self.output_dir, self.log_dir):
            directory.mkdir(exist_ok=True, parents=True)


CONFIG = Config()


def setup_logging(config: Config = CONFIG, level: int = logging.INFO) -> logging.Logger:
    """Console + file logging. Idempotent: safe to call from several modules
    or to re-run in a notebook without stacking duplicate handlers."""
    config.ensure_dirs()

    log = logging.getLogger("rag")
    log.setLevel(level)
    log.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    file_handler = logging.FileHandler(config.log_dir / "rag.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    return log


logger = setup_logging()
