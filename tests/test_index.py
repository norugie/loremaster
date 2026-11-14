"""Tests for embedding and storing chunks. Made-up notes only.

Unit tests use a fake embedder, so they need no Ollama or GPU. The test marked
`integration` calls the real local Ollama:
`docker compose run --rm app pytest -m integration`.
"""

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from loremaster.config import DEFAULT_EMBEDDING_MODEL
from loremaster.index import (
    IndexSettings,
    IndexStore,
    OllamaEmbedder,
    embedding_input,
    index_note,
    normalize,
    sync_index,
)
from loremaster.vault import load_notes

INCLUDE = {"status": ["approved"]}
PREFIX = "search_document: "


class FakeEmbedder:
    """Deterministic 8-number vectors made from a hash of each text."""

    def __init__(self, model: str = "fake-embed") -> None:
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [
            [float(b) + 1.0 for b in hashlib.sha256(t.encode()).digest()[:8]]
            for t in texts
        ]


def make_vault(root: Path) -> Path:
    vault = root / "vault"
    (vault / "Places").mkdir(parents=True)
    (vault / "Places" / "mill.md").write_text(
        "---\nstatus: approved\n---\n"
        "# Old Mill\nThe mill grinds grain.\n"
        "## Wheel\nThe wheel turns by [[River Wey|the river]].\n",
        encoding="utf-8",
    )
    (vault / "gate.md").write_text(
        "---\nstatus: approved\n---\nThe north gate shuts at dusk.\n", encoding="utf-8"
    )
    (vault / "draft.md").write_text("---\nstatus: draft\n---\nNot indexed.\n")
    return vault


def snapshot(folder: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in folder.rglob("*")
        if p.is_file()
    }


def test_indexes_chunks_with_metadata(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    db = tmp_path / "data" / "index.sqlite"
    with IndexStore(db, IndexSettings("fake-embed", PREFIX)) as store:
        stats = sync_index(vault, INCLUDE, store, FakeEmbedder())
        assert (stats.added, stats.chunks_embedded) == (2, 3)
        assert store.count_chunks() == 3
        assert set(store.note_hashes()) == {"Places/mill.md", "gate.md"}
        assert store.meta("embedding_model") == "fake-embed"
        assert store.meta("dimension") == "8"

    rows = sqlite3.connect(db).execute(
        "SELECT note_path, idx, title, headings, text, links, embedding"
        " FROM chunks ORDER BY note_path, idx"
    )
    by_key = {(r[0], r[1]): r for r in rows}
    wheel = by_key[("Places/mill.md", 1)]
    assert wheel[2] == "mill"
    assert json.loads(wheel[3]) == ["Old Mill", "Wheel"]
    assert wheel[4] == "The wheel turns by the river."  # clean text, no prefix
    assert json.loads(wheel[5]) == ["River Wey"]
    vector = np.frombuffer(wheel[6], dtype=np.float32)
    assert vector.shape == (8,)
    assert np.isclose(np.linalg.norm(vector), 1.0)  # stored at length 1


def test_embedding_input_has_prefix_label_and_text(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    fake = FakeEmbedder()
    with IndexStore(tmp_path / "i.sqlite", IndexSettings(fake.model, PREFIX)) as store:
        sync_index(vault, INCLUDE, store, fake)
    sent = [t for call in fake.calls for t in call]
    assert (
        "search_document: mill > Old Mill > Wheel\n\nThe wheel turns by the river."
        in sent
    )
    assert "search_document: gate\n\nThe north gate shuts at dusk." in sent


def test_batches_embed_calls(tmp_path: Path) -> None:
    vault = tmp_path / "v"
    vault.mkdir()
    body = "\n".join(f"## Part {i}\nText {i}." for i in range(5))
    (vault / "n.md").write_text(f"---\nstatus: approved\n---\n{body}\n")
    fake = FakeEmbedder()
    with IndexStore(tmp_path / "i.sqlite", IndexSettings(fake.model, PREFIX)) as store:
        [note] = load_notes(vault, INCLUDE).notes
        index_note(note, store, fake, batch_size=2)
    assert [len(c) for c in fake.calls] == [2, 2, 1]


def test_reindexing_a_note_replaces_its_chunks(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    with IndexStore(
        tmp_path / "i.sqlite", IndexSettings("fake-embed", PREFIX)
    ) as store:
        sync_index(vault, INCLUDE, store, FakeEmbedder())
        old_hash = store.note_hashes()["gate.md"]
        (vault / "gate.md").write_text("---\nstatus: approved\n---\n# A\nx\n# B\ny\n")
        sync_index(vault, INCLUDE, store, FakeEmbedder())
        assert store.count_chunks() == 4
        assert store.note_hashes()["gate.md"] != old_hash


def test_changing_the_model_clears_the_index(tmp_path: Path) -> None:
    db = tmp_path / "i.sqlite"
    with IndexStore(db, IndexSettings("fake-embed", PREFIX)) as store:
        sync_index(make_vault(tmp_path), INCLUDE, store, FakeEmbedder())
    with IndexStore(db, IndexSettings("other-embed", PREFIX)) as store:
        assert store.count_chunks() == 0
        assert store.note_hashes() == {}
        assert store.meta("embedding_model") == "other-embed"
        assert store.meta("dimension") is None


def test_reopening_with_the_same_model_keeps_the_index(tmp_path: Path) -> None:
    db = tmp_path / "i.sqlite"
    with IndexStore(db, IndexSettings("fake-embed", PREFIX)) as store:
        sync_index(make_vault(tmp_path), INCLUDE, store, FakeEmbedder())
    with IndexStore(db, IndexSettings("fake-embed", PREFIX)) as store:
        assert store.count_chunks() == 3


def test_dimension_mismatch_is_an_error(tmp_path: Path) -> None:
    class Wider(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0] * 9 for _ in texts]

    vault = make_vault(tmp_path)
    with IndexStore(
        tmp_path / "i.sqlite", IndexSettings("fake-embed", PREFIX)
    ) as store:
        sync_index(vault, INCLUDE, store, FakeEmbedder())
        with pytest.raises(ValueError, match="embedding size 9"):
            index_note(load_notes(vault, INCLUDE).notes[0], store, Wider())
        assert store.count_chunks() == 3  # the failed note was left as it was


def test_index_inside_the_vault_is_refused(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    with IndexStore(vault / "i.sqlite", IndexSettings("fake-embed", PREFIX)) as store:
        with pytest.raises(ValueError, match="outside the vault"):
            sync_index(vault, INCLUDE, store, FakeEmbedder())


def test_indexing_does_not_change_the_vault(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    before = snapshot(vault)
    with IndexStore(
        tmp_path / "i.sqlite", IndexSettings("fake-embed", PREFIX)
    ) as store:
        sync_index(vault, INCLUDE, store, FakeEmbedder())
    assert snapshot(vault) == before


def test_normalize() -> None:
    out = normalize([[3.0, 4.0], [0.0, 2.0]])
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, 1.0]])
    with pytest.raises(ValueError, match="zero vector"):
        normalize([[0.0, 0.0]])


def test_embedding_input_without_prefix() -> None:
    from loremaster.chunking import Chunk

    chunk = Chunk(Path("a.md"), "a", ("H",), 0, "Text.", ())
    assert embedding_input(chunk, "") == "a > H\n\nText."


@pytest.mark.integration
def test_indexes_a_made_up_vault_through_ollama(tmp_path: Path) -> None:
    """Embed and store through the real local Ollama and embedding model."""
    vault = make_vault(tmp_path)
    before = snapshot(vault)
    embedder = OllamaEmbedder(DEFAULT_EMBEDDING_MODEL)
    with IndexStore(
        tmp_path / "data" / "i.sqlite", IndexSettings(embedder.model, PREFIX)
    ) as store:
        stats = sync_index(vault, INCLUDE, store, embedder)
        assert (stats.added, stats.chunks_embedded) == (2, 3)
        assert store.count_chunks() == 3
        assert store.meta("dimension") == "768"  # nomic-embed-text
    assert snapshot(vault) == before
