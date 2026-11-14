"""Tests for re-indexing on file change. Made-up notes only.

Unit tests drive the change handler with synthetic change sets and a fake
embedder. The test marked `integration` runs the real watcher (watchfiles) on a
temp folder with the real local Ollama.
"""

import asyncio
import hashlib
import logging
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from watchfiles import Change

from loremaster.config import DEFAULT_DOCUMENT_PREFIX, DEFAULT_EMBEDDING_MODEL
from loremaster.index import IndexSettings, IndexStore, OllamaEmbedder, SyncStats
from loremaster.retrieve import Retriever
from loremaster.watcher import VaultSyncer, is_note_change, main, watch_vault

INCLUDE = {"status": ["approved"]}
SETTINGS = IndexSettings("fake-embed", "search_document: ")


class FakeEmbedder:
    def __init__(self) -> None:
        self.model = "fake-embed"
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [
            [float(b) + 1 for b in hashlib.sha256(t.encode()).digest()[:8]]
            for t in texts
        ]


def write(vault: Path, name: str, body: str, status: str = "approved") -> Path:
    path = vault / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nstatus: {status}\n---\n{body}\n", encoding="utf-8")
    return path


def paths(db: Path) -> set[str]:
    return {r[0] for r in sqlite3.connect(db).execute("SELECT path FROM notes")}


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    write(v, "alpha.md", "# Alpha\nFirst.")
    write(v, "beta.md", "# Beta\nSecond.")
    return v


def make_syncer(
    tmp_path: Path,
    vault: Path,
    embedder: FakeEmbedder | None = None,
    on_sync: Callable[[SyncStats], None] | None = None,
) -> VaultSyncer:
    return VaultSyncer(
        vault=vault,
        include=INCLUDE,
        index_path=tmp_path / "data" / "index.sqlite",
        settings=SETTINGS,
        embedder=embedder or FakeEmbedder(),
        on_sync=on_sync,
        retry_delay=0.01,
    )


# --- which paths count -----------------------------------------------------


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("note.md", True),
        ("Sub/Deep/note.md", True),
        ("NOTE.MD", True),
        ("note.md.tmp", False),
        ("image.png", False),
        (".obsidian/workspace.md", False),
        (".trash/old.md", False),
        ("Sub/.hidden.md", False),
    ],
)
def test_is_note_change(vault: Path, rel: str, expected: bool) -> None:
    assert is_note_change(vault, str(vault / rel)) is expected


def test_paths_outside_the_vault_dont_count(tmp_path: Path, vault: Path) -> None:
    assert not is_note_change(vault, str(tmp_path / "elsewhere.md"))
    assert not is_note_change(vault, str(tmp_path / "data" / "index.sqlite"))


# --- handling change sets --------------------------------------------------


def test_irrelevant_changes_do_nothing(tmp_path: Path, vault: Path) -> None:
    embedder = FakeEmbedder()
    syncer = make_syncer(tmp_path, vault, embedder)
    changes = {(Change.modified, str(vault / ".obsidian" / "a.md"))}
    assert asyncio.run(syncer.handle(changes)) is None
    assert embedder.texts == []
    assert not syncer.index_path.exists()


@pytest.mark.parametrize("kind", ["created", "edited", "deleted", "renamed", "moved"])
def test_each_kind_of_change_is_synced(tmp_path: Path, vault: Path, kind: str) -> None:
    embedder = FakeEmbedder()
    syncer = make_syncer(tmp_path, vault, embedder)
    asyncio.run(syncer.sync())
    embedder.texts.clear()

    if kind == "created":
        changes = {(Change.added, str(write(vault, "gamma.md", "New.")))}
        expected = {"alpha.md", "beta.md", "gamma.md"}
    elif kind == "edited":
        changes = {(Change.modified, str(write(vault, "beta.md", "# Beta\nEdited.")))}
        expected = {"alpha.md", "beta.md"}
    elif kind == "deleted":
        (vault / "beta.md").unlink()
        changes = {(Change.deleted, str(vault / "beta.md"))}
        expected = {"alpha.md"}
    elif kind == "renamed":
        (vault / "beta.md").rename(vault / "omega.md")
        changes = {
            (Change.deleted, str(vault / "beta.md")),
            (Change.added, str(vault / "omega.md")),
        }
        expected = {"alpha.md", "omega.md"}
    else:
        (vault / "Moved").mkdir()
        (vault / "beta.md").rename(vault / "Moved" / "beta.md")
        changes = {
            (Change.deleted, str(vault / "beta.md")),
            (Change.added, str(vault / "Moved" / "beta.md")),
        }
        expected = {"alpha.md", "Moved/beta.md"}

    stats = asyncio.run(syncer.handle(changes))
    assert stats is not None
    assert paths(syncer.index_path) == expected
    assert stats.unchanged >= 1
    assert not any("alpha" in t for t in embedder.texts)


def test_filter_changes_follow_frontmatter(tmp_path: Path, vault: Path) -> None:
    syncer = make_syncer(tmp_path, vault)
    asyncio.run(syncer.sync())
    path = write(vault, "alpha.md", "# Alpha\nFirst.", status="draft")
    asyncio.run(syncer.handle({(Change.modified, str(path))}))
    assert paths(syncer.index_path) == {"beta.md"}
    write(vault, "alpha.md", "# Alpha\nFirst.", status="approved")
    asyncio.run(syncer.handle({(Change.modified, str(path))}))
    assert paths(syncer.index_path) == {"alpha.md", "beta.md"}


def test_on_sync_refreshes_the_retriever(tmp_path: Path, vault: Path) -> None:
    embedder = FakeEmbedder()
    syncer = make_syncer(tmp_path, vault, embedder)
    asyncio.run(syncer.sync())
    with IndexStore(syncer.index_path, None) as store:
        retriever = Retriever(store, embedder, "")
        syncer.on_sync = lambda _stats: retriever.refresh()
        assert len(retriever.search("x", k=10)) == 2
        path = write(vault, "gamma.md", "New note.")
        asyncio.run(syncer.handle({(Change.added, str(path))}))
        assert len(retriever.search("x", k=10)) == 3


def test_half_written_note_is_retried(tmp_path: Path, vault: Path) -> None:
    """The first sync sees broken frontmatter; the file is finished before the
    retry, which then indexes it."""
    syncer = make_syncer(tmp_path, vault)
    asyncio.run(syncer.sync())
    half = vault / "gamma.md"
    half.write_text("---\nstatus: appr")  # caught mid-write: unclosed block
    original_sync = syncer.sync
    calls = 0

    async def sync_then_finish() -> SyncStats | None:
        nonlocal calls
        calls += 1
        result = await original_sync()
        if calls == 1:
            write(vault, "gamma.md", "Finished.")
        return result

    syncer.sync = sync_then_finish  # type: ignore[method-assign]
    stats = asyncio.run(syncer.handle({(Change.added, str(half))}))
    assert calls == 2
    assert stats is not None and stats.added == 1
    assert "gamma.md" in paths(syncer.index_path)


def test_note_broken_for_good_is_retried_only_once(tmp_path: Path, vault: Path) -> None:
    """A note that is skipped on every sync must not double every later sync."""
    syncs = 0

    def count(_stats: SyncStats) -> None:
        nonlocal syncs
        syncs += 1

    syncer = make_syncer(tmp_path, vault, on_sync=count)
    broken = vault / "gamma.md"
    broken.write_text("---\nstatus: appr")  # unclosed frontmatter, never fixed
    asyncio.run(syncer.handle({(Change.added, str(broken))}))
    assert syncs == 2  # the first time it might be half-written: retried
    asyncio.run(syncer.handle({(Change.modified, str(vault / "alpha.md"))}))
    assert syncs == 3  # still broken, count unchanged: no retry


def test_failed_sync_is_logged_and_does_not_raise(
    tmp_path: Path, vault: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class Down(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise ConnectionError("Ollama is not running")

    syncer = make_syncer(tmp_path, vault, Down())
    with caplog.at_level(logging.ERROR):
        result = asyncio.run(syncer.handle({(Change.added, str(vault / "alpha.md"))}))
    assert result is None
    assert "index sync failed" in caplog.text


def test_failing_on_sync_is_logged_and_does_not_raise(
    tmp_path: Path, vault: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An error in the callback must not end the watcher task."""

    def broken(_stats: SyncStats) -> None:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "half-saved guidance note")

    syncer = make_syncer(tmp_path, vault, on_sync=broken)
    with caplog.at_level(logging.ERROR):
        assert asyncio.run(syncer.sync()) is None
    assert "index sync failed" in caplog.text


def test_sync_runs_off_the_event_loop(tmp_path: Path, vault: Path) -> None:
    import threading

    seen: list[str] = []

    class Recording(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            seen.append(threading.current_thread().name)
            return super().embed(texts)

    syncer = make_syncer(tmp_path, vault, Recording())
    asyncio.run(syncer.sync())
    assert seen and threading.main_thread().name not in seen


def test_watch_stops_cleanly_on_cancel(tmp_path: Path, vault: Path) -> None:
    syncer = make_syncer(tmp_path, vault)

    async def scenario() -> None:
        task = asyncio.create_task(watch_vault(syncer))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert paths(syncer.index_path) == {"alpha.md", "beta.md"}  # initial sync ran


def test_watch_stops_on_stop_event(tmp_path: Path, vault: Path) -> None:
    syncer = make_syncer(tmp_path, vault)

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(watch_vault(syncer, stop_event=stop))
        await asyncio.sleep(0.5)
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())


def test_main_needs_a_config(tmp_path: Path) -> None:
    assert main(["--config", str(tmp_path / "none.yaml")]) == 1


# --- the real watcher ------------------------------------------------------


@pytest.mark.integration
def test_index_follows_changes_while_the_watcher_runs(
    tmp_path: Path, vault: Path
) -> None:
    """TASKS 3.5 Verify: add, edit, rename and delete made-up notes while the
    watcher runs; the index follows each change."""
    syncer = VaultSyncer(
        vault=vault,
        include=INCLUDE,
        index_path=tmp_path / "data" / "index.sqlite",
        settings=IndexSettings(DEFAULT_EMBEDDING_MODEL, DEFAULT_DOCUMENT_PREFIX),
        embedder=OllamaEmbedder(DEFAULT_EMBEDDING_MODEL),
    )
    db = syncer.index_path

    def texts() -> set[str]:
        return {r[0] for r in sqlite3.connect(db).execute("SELECT text FROM chunks")}

    async def until(check: Callable[[], bool], timeout: float = 20) -> None:
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while not check():
            if loop.time() > end:
                raise AssertionError("index did not follow the change in time")
            await asyncio.sleep(0.1)

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(watch_vault(syncer, stop_event=stop))
        await until(lambda: db.exists() and paths(db) == {"alpha.md", "beta.md"})
        await asyncio.sleep(0.3)  # let the watcher start

        write(vault, "gamma.md", "A new note about a windmill.")
        await until(lambda: "gamma.md" in paths(db))

        await asyncio.sleep(1.1)  # polling sees whole-second mtimes only
        write(vault, "beta.md", "# Beta\nSecond, edited.")
        await until(lambda: "Second, edited." in texts())

        (vault / "alpha.md").rename(vault / "omega.md")
        await until(lambda: paths(db) == {"beta.md", "gamma.md", "omega.md"})

        (vault / "gamma.md").unlink()
        await until(lambda: paths(db) == {"beta.md", "omega.md"})

        stop.set()
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())
