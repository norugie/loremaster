"""Re-index the vault when its notes change, while the bot runs.

`watchfiles.awatch` reports file changes in batches: it waits until the folder
has been quiet for `DEBOUNCE_MS`, so the burst of events from one save (or from
an editor that writes a temp file and renames it) arrives as one batch. For
each batch that touches a `.md` note outside dot-folders, the same
`sync_index` as at startup runs, so there is one code path for every
kind of change: created, edited, deleted, renamed, moved, or a note whose
frontmatter starts or stops passing the filter. Syncing hashes every note but
only embeds the changed ones (about 0.1 s for a ~130-note vault when nothing
changed).

Embedding is slow and blocking, so each sync runs in a worker thread
(`asyncio.to_thread`) and never blocks the bot's event loop. The worker opens
its own `IndexStore`, because a SQLite connection belongs to the thread that
made it.

A note caught half-written or locked is skipped by the sync (its old indexed
version is kept); the watcher then tries once more after `RETRY_DELAY_S` (only
when more notes were skipped than last time, so a note that stays broken isn't
retried on every save), and the save that finishes the file triggers another
batch anyway. A failed sync is
logged and the watcher keeps going. Cancelling the task, or setting
`stop_event`, stops it cleanly.

Try it by hand, with the bot's config (it never writes to the vault):

    docker compose run --rm app python -m loremaster.watcher

(`--config PATH` for a config other than ./lorebot.yaml)
"""

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from watchfiles import Change, awatch

from loremaster.config import ConfigError, load_config
from loremaster.index import (
    Embedder,
    IndexSettings,
    IndexStore,
    OllamaEmbedder,
    SyncStats,
    sync_index,
)

log = logging.getLogger(__name__)

DEBOUNCE_MS = 500
RETRY_DELAY_S = 2.0

FileChanges = Iterable[tuple[Change, str]]


def is_note_change(vault: Path, path: str) -> bool:
    """Whether a changed path can affect the index: a `.md` file inside the
    vault and outside any folder or file starting with "."."""
    p = Path(path)
    if p.suffix.lower() != ".md":
        return False
    try:
        rel = p.resolve().relative_to(vault.resolve())
    except ValueError:
        return False
    return not any(part.startswith(".") for part in rel.parts)


class VaultSyncer:
    """Runs `sync_index` for batches of file changes, off the event loop."""

    def __init__(
        self,
        vault: Path,
        include: Mapping[str, Sequence[str]],
        index_path: Path,
        settings: IndexSettings,
        embedder: Embedder,
        on_sync: Callable[[SyncStats], None] | None = None,
        retry_delay: float = RETRY_DELAY_S,
    ) -> None:
        self.vault = vault
        self.include = include
        self.index_path = index_path
        self.settings = settings
        self.embedder = embedder
        self.on_sync = on_sync  # e.g. refresh the retriever
        self.retry_delay = retry_delay
        self.last_skipped = 0  # skipped notes in the last handled sync

    async def sync(self) -> SyncStats | None:
        """Run one sync in a worker thread. Returns None if it failed."""
        try:
            stats = await asyncio.to_thread(self.sync_blocking)
            if self.on_sync is not None:
                self.on_sync(stats)
        except Exception:
            # Never let one failed sync (e.g. Ollama briefly down) stop the
            # watcher; the next change tries again.
            log.exception("index sync failed")
            return None
        return stats

    async def handle(self, changes: FileChanges) -> SyncStats | None:
        """React to one batch of changes. Returns the last sync's stats, or None
        if nothing relevant changed or the sync failed."""
        relevant = sorted(p for _, p in changes if is_note_change(self.vault, p))
        if not relevant:
            return None
        log.info("vault changed: %s", ", ".join(Path(p).name for p in relevant[:5]))
        stats = await self.sync()
        if stats is not None and stats.skipped > self.last_skipped:
            # A note may have been caught half-written or locked: try once more.
            # (Not when the count didn't rise: a note that is broken for good
            # would otherwise double every sync.)
            await asyncio.sleep(self.retry_delay)
            stats = await self.sync() or stats
        if stats is not None:
            self.last_skipped = stats.skipped
        return stats

    def sync_blocking(self) -> SyncStats:
        with IndexStore(self.index_path, self.settings) as store:
            return sync_index(self.vault, self.include, store, self.embedder)


async def watch_vault(
    syncer: VaultSyncer,
    stop_event: asyncio.Event | None = None,
    debounce_ms: int = DEBOUNCE_MS,
    initial_sync: bool = True,
) -> None:
    """Sync once (skipped if `initial_sync` is False), then keep the index in
    line with the vault until cancelled or
    `stop_event` is set."""
    if initial_sync:
        await syncer.sync()
    log.info("watching %s for changes", syncer.vault)
    async for changes in awatch(
        syncer.vault,
        watch_filter=lambda _change, path: is_note_change(syncer.vault, path),
        debounce=debounce_ms,
        stop_event=stop_event,
    ):
        await syncer.handle(changes)
    log.info("stopped watching %s", syncer.vault)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m loremaster.watcher",
        description="Keep the index in line with a vault, logging one line per sync.",
    )
    parser.add_argument("--config", type=Path, default=Path("lorebot.yaml"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    syncer = VaultSyncer(
        vault=config.vault_path,
        include=config.include,
        index_path=config.index_path,
        settings=IndexSettings(config.models.embedding, config.models.document_prefix),
        embedder=OllamaEmbedder(config.models.embedding),
    )
    try:
        asyncio.run(_watch_until_signalled(syncer))
    except KeyboardInterrupt:
        pass
    return 0


async def _watch_until_signalled(syncer: VaultSyncer) -> None:
    """Watch until Ctrl+C or, in a container, `docker stop` (SIGTERM)."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows event loops don't support this
            pass
    await watch_vault(syncer, stop_event=stop)


if __name__ == "__main__":
    sys.exit(main())
