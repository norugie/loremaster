"""Tests for the incremental index sync. Made-up notes only.

A counting fake embedder shows exactly which notes get re-embedded. The test
marked `integration` repeats the main check through the real local Ollama.
"""

import hashlib
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import ollama
import pytest

from loremaster.config import DEFAULT_DOCUMENT_PREFIX, DEFAULT_EMBEDDING_MODEL
from loremaster.index import IndexSettings, IndexStore, OllamaEmbedder, sync_index

INCLUDE = {"status": ["approved"]}
SETTINGS = IndexSettings("fake-embed", "search_document: ")


class CountingEmbedder:
    """Fake embedder that records every text it was asked to embed."""

    def __init__(self, model: str = "fake-embed") -> None:
        self.model = model
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [
            [float(b) + 1 for b in hashlib.sha256(t.encode()).digest()[:8]]
            for t in texts
        ]

    def notes_embedded(self) -> set[str]:
        """Titles of notes whose chunks were embedded (from the label)."""
        return {t.split(": ", 1)[1].split(" > ")[0].split("\n")[0] for t in self.texts}


def write(vault: Path, name: str, body: str, status: str = "approved") -> None:
    path = vault / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nstatus: {status}\n---\n{body}\n", encoding="utf-8")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    write(v, "alpha.md", "# Alpha\nFirst note.")
    write(v, "beta.md", "# Beta\nSecond note.\n## Part\nMore.")
    write(v, "Sub/gamma.md", "Third note, no headings.")
    return v


def sync(
    db: Path,
    vault: Path,
    include: dict[str, list[str]] = INCLUDE,
    settings: IndexSettings = SETTINGS,
) -> tuple[CountingEmbedder, dict[str, int]]:
    """Open the index fresh (like a restart), sync, and return what happened."""
    embedder = CountingEmbedder(settings.model)
    with IndexStore(db, settings) as store:
        stats = sync_index(vault, include, store, embedder)
    counts = {
        "added": stats.added,
        "updated": stats.updated,
        "removed": stats.removed,
        "unchanged": stats.unchanged,
        "skipped": stats.skipped,
        "chunks": stats.chunks_embedded,
    }
    return embedder, counts


def stored_paths(db: Path) -> set[str]:
    return {r[0] for r in sqlite3.connect(db).execute("SELECT path FROM notes")}


def stored_titles(db: Path) -> set[str]:
    return {r[0] for r in sqlite3.connect(db).execute("SELECT title FROM chunks")}


def test_first_sync_embeds_everything(tmp_path: Path, vault: Path) -> None:
    embedder, counts = sync(tmp_path / "i.sqlite", vault)
    assert counts == {
        "added": 3,
        "updated": 0,
        "removed": 0,
        "unchanged": 0,
        "skipped": 0,
        "chunks": 4,
    }
    assert len(embedder.texts) == 4


def test_unchanged_vault_embeds_nothing(tmp_path: Path, vault: Path) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    embedder, counts = sync(db, vault)
    assert embedder.texts == []
    assert counts["unchanged"] == 3
    assert counts["chunks"] == 0


def test_edited_note_is_reembedded_alone(tmp_path: Path, vault: Path) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    write(vault, "beta.md", "# Beta\nSecond note, edited.")
    embedder, counts = sync(db, vault)
    assert embedder.notes_embedded() == {"beta"}
    assert (counts["updated"], counts["unchanged"], counts["chunks"]) == (1, 2, 1)


def test_frontmatter_only_change_is_not_reembedded(tmp_path: Path, vault: Path) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    (vault / "alpha.md").write_text(
        "---\nstatus: approved\naliases: [A]\n---\n# Alpha\nFirst note.\n"
    )
    embedder, counts = sync(db, vault)
    assert embedder.texts == []  # frontmatter isn't embedded
    assert counts["unchanged"] == 3


def test_new_note_is_added(tmp_path: Path, vault: Path) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    write(vault, "delta.md", "Fourth note.")
    embedder, counts = sync(db, vault)
    assert embedder.notes_embedded() == {"delta"}
    assert counts["added"] == 1
    assert "delta.md" in stored_paths(db)


def test_deleted_note_is_removed_with_its_chunks(tmp_path: Path, vault: Path) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    (vault / "beta.md").unlink()
    embedder, counts = sync(db, vault)
    assert embedder.texts == []
    assert counts["removed"] == 1
    assert "beta.md" not in stored_paths(db)
    assert "beta" not in stored_titles(db)


def test_renamed_note_is_reembedded_under_its_new_title(
    tmp_path: Path, vault: Path
) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    (vault / "alpha.md").rename(vault / "omega.md")  # same body, new title
    embedder, counts = sync(db, vault)
    assert embedder.notes_embedded() == {"omega"}
    assert (counts["added"], counts["removed"]) == (1, 1)
    assert "alpha" not in stored_titles(db)
    assert "omega" in stored_titles(db)


def test_moved_note_is_reindexed_at_its_new_path(tmp_path: Path, vault: Path) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    (vault / "Other").mkdir()
    (vault / "Sub" / "gamma.md").rename(vault / "Other" / "gamma.md")
    _, counts = sync(db, vault)
    assert (counts["added"], counts["removed"]) == (1, 1)
    assert "Other/gamma.md" in stored_paths(db)
    assert "Sub/gamma.md" not in stored_paths(db)


def test_note_that_stops_passing_the_filter_is_removed(
    tmp_path: Path, vault: Path
) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    write(vault, "alpha.md", "# Alpha\nFirst note.", status="draft")
    embedder, counts = sync(db, vault)
    assert embedder.texts == []
    assert counts["removed"] == 1
    assert "alpha.md" not in stored_paths(db)


def test_note_that_starts_passing_the_filter_is_added(
    tmp_path: Path, vault: Path
) -> None:
    db = tmp_path / "i.sqlite"
    write(vault, "draft.md", "Draft text.", status="draft")
    sync(db, vault)
    assert "draft.md" not in stored_paths(db)
    write(vault, "draft.md", "Draft text.", status="approved")
    embedder, counts = sync(db, vault)
    assert embedder.notes_embedded() == {"draft"}
    assert counts["added"] == 1


def test_changed_include_filter(tmp_path: Path, vault: Path) -> None:
    db = tmp_path / "i.sqlite"
    write(vault, "draft.md", "Draft text.", status="draft")
    sync(db, vault)
    embedder, counts = sync(db, vault, include={"status": ["approved", "draft"]})
    assert embedder.notes_embedded() == {"draft"}
    assert (counts["added"], counts["unchanged"]) == (1, 3)
    embedder, counts = sync(db, vault, include={"status": ["draft"]})
    assert embedder.texts == []
    assert (counts["removed"], counts["unchanged"]) == (3, 1)


@pytest.mark.parametrize(
    "changed",
    [
        IndexSettings("fake-embed", "other prefix: "),
        IndexSettings("fake-embed", "search_document: ", max_chunk_chars=500),
        IndexSettings("fake-embed", "search_document: ", chunker_version="test"),
        IndexSettings("other-embed", "search_document: "),
    ],
)
def test_changed_index_settings_rebuild_everything(
    tmp_path: Path, vault: Path, changed: IndexSettings
) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    embedder, counts = sync(db, vault, settings=changed)
    assert counts["added"] == 3
    assert len(embedder.texts) == 4
    if changed.document_prefix != SETTINGS.document_prefix:
        assert all(t.startswith("other prefix: ") for t in embedder.texts)


def test_broken_note_keeps_its_old_version_and_does_not_stop_sync(
    tmp_path: Path, vault: Path, caplog: pytest.LogCaptureFixture
) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    (vault / "alpha.md").write_text("---\nstatus: [oops\n---\n# Alpha\nBroken.\n")
    write(vault, "beta.md", "# Beta\nEdited.")
    with caplog.at_level(logging.WARNING):
        embedder, counts = sync(db, vault)
    assert embedder.notes_embedded() == {"beta"}  # the rest still synced
    assert (counts["skipped"], counts["removed"], counts["updated"]) == (1, 0, 1)
    assert "alpha.md" in stored_paths(db)  # old version kept
    assert "alpha.md" in caplog.text


def test_embedding_failure_skips_note_and_keeps_old_version(
    tmp_path: Path, vault: Path
) -> None:
    class FailsOnBeta(CountingEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            if any("beta" in t for t in texts):
                raise ollama.ResponseError("input too long")
            return super().embed(texts)

    db = tmp_path / "i.sqlite"
    sync(db, vault)
    write(vault, "alpha.md", "# Alpha\nEdited.")
    write(vault, "beta.md", "# Beta\nEdited too.")
    with IndexStore(db, SETTINGS) as store:
        stats = sync_index(vault, INCLUDE, store, FailsOnBeta())
        assert (stats.updated, stats.skipped) == (1, 1)
        chunks, _ = store.all_chunks()
    beta_text = {c.text for c in chunks if c.title == "beta"}
    assert beta_text == {"Second note.", "More."}  # old version, untouched


def test_missing_embedding_model_stops_the_sync(tmp_path: Path, vault: Path) -> None:
    """Ollama's "model not found" (404) would fail every note: fail loudly."""

    class NotPulled(CountingEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise ollama.ResponseError("model 'fake-embed' not found", 404)

    with IndexStore(tmp_path / "i.sqlite", SETTINGS) as store:
        with pytest.raises(ollama.ResponseError, match="not found"):
            sync_index(vault, INCLUDE, store, NotPulled())


def test_every_note_failing_stops_the_sync(tmp_path: Path, vault: Path) -> None:
    """With nothing indexed and every embed failing, Ollama is broken: fail
    loudly instead of starting with an empty index."""

    class Broken(CountingEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise ollama.ResponseError("model failed to load", 500)

    with IndexStore(tmp_path / "i.sqlite", SETTINGS) as store:
        with pytest.raises(ollama.ResponseError, match="failed to load"):
            sync_index(vault, INCLUDE, store, Broken())


def test_empty_vault_does_not_wipe_the_index(tmp_path: Path, vault: Path) -> None:
    db = tmp_path / "i.sqlite"
    sync(db, vault)
    vault.rename(tmp_path / "moved")  # e.g. the folder is briefly gone
    vault.mkdir()
    with pytest.raises(ValueError, match="no notes found"):
        sync(db, vault)
    assert len(stored_paths(db)) == 3


def test_dense_note_is_retried_with_smaller_chunks(tmp_path: Path, vault: Path) -> None:
    """Text with many tokens per character overflows the embedder at the usual
    chunk size; the note is embedded in half-size chunks instead of skipped."""

    class SmallContext(CountingEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            if any(len(t) > 1200 for t in texts):
                raise ollama.ResponseError("input length exceeds context", 400)
            return super().embed(texts)

    write(vault, "dense.md", " ".join(["word"] * 400))  # one 2,000-character chunk
    embedder = SmallContext()
    with IndexStore(tmp_path / "i.sqlite", SETTINGS) as store:
        stats = sync_index(vault, INCLUDE, store, embedder)
        chunks, _ = store.all_chunks()
    assert stats.skipped == 0
    dense = [c.text for c in chunks if c.title == "dense"]
    assert len(dense) == 2 and " ".join(dense) == " ".join(["word"] * 400)


def test_failure_mid_note_leaves_the_note_whole(tmp_path: Path, vault: Path) -> None:
    """Embedding fails after the first batch: nothing of the note is written."""

    class FailsSecondCall(CountingEmbedder):
        calls = 0

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls += 1
            if self.calls >= 2:
                raise ollama.ResponseError("boom")
            return super().embed(texts)

    db = tmp_path / "i.sqlite"
    sync(db, vault)
    write(vault, "beta.md", "# Beta\nNew one.\n## A\nx\n## B\ny")
    from loremaster.index import index_note
    from loremaster.vault import load_notes

    beta = next(n for n in load_notes(vault, INCLUDE).notes if n.path.stem == "beta")
    with IndexStore(db, SETTINGS) as store:
        with pytest.raises(ollama.ResponseError):
            index_note(beta, store, FailsSecondCall(), batch_size=1)
        chunks, _ = store.all_chunks()
    assert {c.text for c in chunks if c.title == "beta"} == {"Second note.", "More."}


def test_sync_logs_a_summary(
    tmp_path: Path, vault: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="loremaster.index"):
        sync(tmp_path / "i.sqlite", vault)
    assert "index sync: 3 added, 0 updated, 0 removed, 0 unchanged, 0 skipped" in (
        caplog.text
    )


def test_wrong_embedder_or_read_only_store_is_refused(
    tmp_path: Path, vault: Path
) -> None:
    db = tmp_path / "i.sqlite"
    with IndexStore(db, SETTINGS) as store:
        with pytest.raises(ValueError, match="doesn't match"):
            sync_index(vault, INCLUDE, store, CountingEmbedder("other"))
    with IndexStore(db, None) as store:
        with pytest.raises(ValueError, match="searching only"):
            sync_index(vault, INCLUDE, store, CountingEmbedder())


@pytest.mark.integration
def test_restart_reembeds_only_the_changed_note_through_ollama(
    tmp_path: Path, vault: Path
) -> None:
    """TASKS 3.4 Verify: index a made-up vault, change one note, restart."""

    class Counting(OllamaEmbedder):
        def __init__(self, model: str) -> None:
            super().__init__(model)
            self.texts: list[str] = []

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.texts.extend(texts)
            return super().embed(texts)

    db = tmp_path / "data" / "i.sqlite"
    settings = IndexSettings(DEFAULT_EMBEDDING_MODEL, DEFAULT_DOCUMENT_PREFIX)

    first = Counting(DEFAULT_EMBEDDING_MODEL)
    with IndexStore(db, settings) as store:
        assert sync_index(vault, INCLUDE, store, first).added == 3
    assert len(first.texts) == 4

    write(vault, "beta.md", "# Beta\nSecond note, edited.")
    second = Counting(DEFAULT_EMBEDDING_MODEL)  # a fresh process, in effect
    with IndexStore(db, settings) as store:
        stats = sync_index(vault, INCLUDE, store, second)
    assert (stats.updated, stats.unchanged) == (1, 2)
    assert second.texts == ["search_document: beta > Beta\n\nSecond note, edited."]
