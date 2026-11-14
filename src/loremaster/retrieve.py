"""Find the chunks most similar to a question (brute-force cosine similarity).

The question is embedded with the embedding model's query prefix (for
nomic-embed-text, `search_query: `), scaled to length 1, and compared with
every stored vector. Stored vectors are already length 1 (see index.py), so a
dot product is the cosine similarity: 1.0 means pointing the same way, 0 means
unrelated. Scores are returned as they are; deciding what counts as "relevant
enough" is left to the answer step, with a threshold from the config
(`retrieval.min_score`).

Try it by hand, after an index has been built:

    docker compose run --rm app python -m loremaster.retrieve "your question"
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from loremaster.chunking import Chunk
from loremaster.config import ConfigError, load_config
from loremaster.index import Embedder, IndexStore, OllamaEmbedder, normalize

DEFAULT_TOP_K = 5


class IndexMismatchError(Exception):
    """The index was built with a different embedding model or vector size."""


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk and its similarity to the question."""

    chunk: Chunk
    score: float  # cosine similarity, from -1 to 1; higher is more similar

    @property
    def label(self) -> str:
        return self.chunk.label


class Retriever:
    """Searches an index. Chunks and vectors are read from the database once
    and kept in memory; call `refresh()` after the index changes."""

    def __init__(
        self, store: IndexStore, embedder: Embedder, query_prefix: str
    ) -> None:
        stored_model = store.meta("embedding_model")
        if stored_model != embedder.model:
            raise IndexMismatchError(
                f"index was built with {stored_model!r}, but the configured "
                f"embedding model is {embedder.model!r}; rebuild the index"
            )
        self._store = store
        self._embedder = embedder
        self._query_prefix = query_prefix
        # Chunks and vectors are swapped together as one tuple, so a search
        # running in a worker thread never sees new chunks with old vectors.
        self._data: tuple[list[Chunk], NDArray[np.float32]] = (
            [],
            np.zeros((0, 0), dtype=np.float32),
        )
        self.refresh()

    def refresh(self) -> None:
        self._data = self._store.all_chunks()

    def search(self, question: str, k: int = DEFAULT_TOP_K) -> list[Hit]:
        """The `k` chunks most similar to `question`, best first.

        Returns fewer than `k` hits if the index is smaller, and none if it's
        empty. Raises `ValueError` for an empty question or `k` below 1.
        """
        if not question.strip():
            raise ValueError("question is empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        chunks, vectors = self._data
        if not chunks:
            return []

        query = normalize(self._embedder.embed([self._query_prefix + question.strip()]))
        if query.shape[1] != vectors.shape[1]:
            raise IndexMismatchError(
                f"question vector has {query.shape[1]} numbers, "
                f"the index has {vectors.shape[1]}; rebuild the index"
            )
        scores = vectors @ query[0]  # one dot product per stored chunk
        k = min(k, len(scores))
        # argpartition finds the top k without sorting everything; then sort those.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(chunk=chunks[i], score=float(scores[i])) for i in top]


def main(argv: list[str] | None = None) -> int:
    """Command-line try-it: print the top chunks' labels and scores."""
    parser = argparse.ArgumentParser(
        prog="python -m loremaster.retrieve",
        description="Show the chunks most similar to a question.",
    )
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="number of hits")
    parser.add_argument("--config", type=Path, default=Path("lorebot.yaml"))
    parser.add_argument("--text", action="store_true", help="also print chunk text")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    if not config.index_path.is_file():
        print(f"no index at {config.index_path}; build one first", file=sys.stderr)
        return 1
    embedder = OllamaEmbedder(config.models.embedding)
    with IndexStore(config.index_path, None) as store:
        retriever = Retriever(store, embedder, config.models.query_prefix)
        start = time.perf_counter()
        hits = retriever.search(args.question, args.k)
        took = time.perf_counter() - start
    for rank, hit in enumerate(hits, start=1):
        print(
            f"{rank}. {hit.score:.3f}  {hit.label}  ({hit.chunk.note_path.as_posix()})"
        )
        if args.text:
            print(f"   {hit.chunk.text}\n")
    print(f"{len(hits)} hits in {took * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
