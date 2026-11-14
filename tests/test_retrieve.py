"""Tests for retrieval. Made-up notes only.

Unit tests use a bag-of-words fake embedder: texts sharing words get similar
vectors, so an obvious question finds its note without Ollama. The test marked
`integration` runs the same check through the real local Ollama.
"""

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from loremaster.config import (
    DEFAULT_DOCUMENT_PREFIX,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_PREFIX,
)
from loremaster.index import IndexSettings, IndexStore, OllamaEmbedder, sync_index
from loremaster.retrieve import IndexMismatchError, Retriever, main

INCLUDE = {"status": ["approved"]}
DIM = 64


class WordsEmbedder:
    """Each word adds 1 to one of 64 slots, picked by a hash of the word."""

    def __init__(self, model: str = "words", dim: int = DIM) -> None:
        self.model = model
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in re.findall(r"[a-z]+", text.lower()):
                vector[hashlib.sha256(word.encode()).digest()[0] % self.dim] += 1.0
            vector[0] += 0.01  # never a zero vector
            out.append(vector)
        return out


NOTES = {
    "bakery.md": "# Bakery\nThe bakery sells rye bread and honey cakes every morning.",
    "lighthouse.md": "# Lighthouse\nThe lighthouse keeper lights the lamp at dusk.",
    "orchard.md": "# Orchard\nApple trees grow in the orchard on the hill.\n"
    "## Harvest\nPickers harvest apples in autumn.",
}


def build(tmp_path: Path, notes: dict[str, str], embedder: WordsEmbedder) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, body in notes.items():
        (vault / name).write_text(f"---\nstatus: approved\n---\n{body}\n")
    db = tmp_path / "data" / "index.sqlite"
    with IndexStore(
        db, IndexSettings(embedder.model, DEFAULT_DOCUMENT_PREFIX)
    ) as store:
        sync_index(vault, INCLUDE, store, embedder)
    return db


def test_obvious_question_finds_its_note(tmp_path: Path) -> None:
    embedder = WordsEmbedder()
    db = build(tmp_path, NOTES, embedder)
    with IndexStore(db, None) as store:
        hits = Retriever(store, embedder, DEFAULT_QUERY_PREFIX).search(
            "Who lights the lighthouse lamp?", k=2
        )
    assert hits[0].chunk.note_path == Path("lighthouse.md")
    assert hits[0].label == "lighthouse > Lighthouse"
    assert hits[0].chunk.text == "The lighthouse keeper lights the lamp at dusk."
    assert hits[0].score > hits[1].score


def test_results_are_sorted_and_scores_are_cosines(tmp_path: Path) -> None:
    embedder = WordsEmbedder()
    db = build(tmp_path, NOTES, embedder)
    with IndexStore(db, None) as store:
        hits = Retriever(store, embedder, DEFAULT_QUERY_PREFIX).search("apples", k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 + 1e-6 for s in scores)
    assert hits[0].label == "orchard > Orchard > Harvest"


def test_question_uses_the_query_prefix(tmp_path: Path) -> None:
    embedder = WordsEmbedder()
    db = build(tmp_path, NOTES, embedder)
    embedder.calls.clear()
    with IndexStore(db, None) as store:
        Retriever(store, embedder, "search_query: ").search("  bread?  ")
    assert embedder.calls == [["search_query: bread?"]]


def test_k_larger_than_index(tmp_path: Path) -> None:
    embedder = WordsEmbedder()
    db = build(tmp_path, NOTES, embedder)
    with IndexStore(db, None) as store:
        hits = Retriever(store, embedder, "").search("bread", k=50)
    assert len(hits) == 4  # 3 notes, 4 chunks


def test_empty_index_returns_nothing(tmp_path: Path) -> None:
    embedder = WordsEmbedder()
    db = build(tmp_path, {}, embedder)
    with IndexStore(db, None) as store:
        assert Retriever(store, embedder, "").search("anything") == []


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_an_error(tmp_path: Path, question: str) -> None:
    embedder = WordsEmbedder()
    db = build(tmp_path, NOTES, embedder)
    with IndexStore(db, None) as store:
        with pytest.raises(ValueError, match="empty"):
            Retriever(store, embedder, "").search(question)


def test_k_below_one_is_an_error(tmp_path: Path) -> None:
    embedder = WordsEmbedder()
    db = build(tmp_path, NOTES, embedder)
    with IndexStore(db, None) as store:
        with pytest.raises(ValueError, match="k must"):
            Retriever(store, embedder, "").search("bread", k=0)


def test_different_model_is_refused(tmp_path: Path) -> None:
    db = build(tmp_path, NOTES, WordsEmbedder(model="words"))
    with IndexStore(db, None) as store:
        with pytest.raises(IndexMismatchError, match="rebuild"):
            Retriever(store, WordsEmbedder(model="other"), "")
        assert store.count_chunks() == 4  # opening for search didn't clear it


def test_different_dimension_is_refused(tmp_path: Path) -> None:
    db = build(tmp_path, NOTES, WordsEmbedder())
    with IndexStore(db, None) as store:
        retriever = Retriever(store, WordsEmbedder(dim=32), "")
        with pytest.raises(IndexMismatchError, match="32 numbers"):
            retriever.search("bread")


def test_refresh_sees_new_chunks(tmp_path: Path) -> None:
    embedder = WordsEmbedder()
    db = build(tmp_path, {"bakery.md": NOTES["bakery.md"]}, embedder)
    with IndexStore(
        db, IndexSettings(embedder.model, DEFAULT_DOCUMENT_PREFIX)
    ) as store:
        retriever = Retriever(store, embedder, "")
        assert len(retriever.search("x", k=10)) == 1
        vault = tmp_path / "vault"
        (vault / "mill.md").write_text("---\nstatus: approved\n---\nThe mill.\n")
        sync_index(vault, INCLUDE, store, embedder)
        assert len(retriever.search("x", k=10)) == 1  # cached until refresh
        retriever.refresh()
        assert len(retriever.search("x", k=10)) == 2


def test_cli_without_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "vault").mkdir()
    config = tmp_path / "lorebot.yaml"
    config.write_text(
        "vault_path: vault\ninclude:\n  status: [approved]\nindex_path: none.sqlite\n",
        encoding="utf-8",
    )
    assert main(["question", "--config", str(config)]) == 1
    assert "no index" in capsys.readouterr().err


@pytest.mark.integration
def test_obvious_question_finds_its_note_through_ollama(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, body in NOTES.items():
        (vault / name).write_text(f"---\nstatus: approved\n---\n{body}\n")
    embedder = OllamaEmbedder(DEFAULT_EMBEDDING_MODEL)
    db = tmp_path / "data" / "index.sqlite"
    with IndexStore(
        db, IndexSettings(embedder.model, DEFAULT_DOCUMENT_PREFIX)
    ) as store:
        sync_index(vault, INCLUDE, store, embedder)
        retriever = Retriever(store, embedder, DEFAULT_QUERY_PREFIX)
        assert retriever.search("Who lights the lamp at night?")[0].label == (
            "lighthouse > Lighthouse"
        )
        assert retriever.search("Where can I buy bread?")[0].label == (
            "bakery > Bakery"
        )
        assert retriever.search("When are apples picked?")[0].label == (
            "orchard > Orchard > Harvest"
        )
