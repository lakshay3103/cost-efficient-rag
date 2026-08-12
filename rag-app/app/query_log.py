"""
Structured per-query logging. Writes one JSON line per query to a log
file plus stdout, capturing exactly what the assignment asks for:
latency, chunk count, and token usage.
"""
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LOG_PATH = Path("logs/queries.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)

logger = logging.getLogger("rag_query_log")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


@dataclass
class QueryLogEntry:
    timestamp: float
    question: str
    top_k: int
    retrieved_chunk_count: int
    relevant_chunk_count: int
    grounded: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def log_query(entry: QueryLogEntry) -> None:
    line = json.dumps(asdict(entry))
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    logger.info(line)


class Timer:
    """Small helper for measuring elapsed wall-clock time in milliseconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
