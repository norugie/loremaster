"""Embed chunks with Ollama and keep them in a SQLite index.

What gets embedded: the embedding model's document prefix, then the chunk's
label (`note title > heading > subheading`), then the chunk text. The label
gives short chunks context. The stored `text` is the clean chunk alone, for the
answer prompt later.

Vectors are stored as float32 bytes, scaled to length 1 when written, so the
search can use a plain dot product as cosine similarity.

Keeping the index up to date:
- Each note is stored under its vault-relative path with a hash of its body.
  On sync, a note is re-embedded only if it is new or its body hash changed.
  The title in each label comes from the path, so a renamed or moved note is a
  new path: the old one is removed and the new one embedded.
- The index also stores a fingerprint of everything else that decides what
  gets embedded (`IndexSettings`: embedding model, document prefix, chunk size,
  chunker version, schema version). If any of those change, the whole index is
  emptied and rebuilt, since old and new vectors can't be mixed.
- Each note is replaced in one SQLite transaction, so a crash never leaves a
  note half indexed.
"""

import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import ollama
from numpy.typing import NDArray

from loremaster.chunking import CHUNKER_VERSION, MAX_CHUNK_CHARS, Chunk, chunk_note
from loremaster.vault import Note, load_notes

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
EMBED_BATCH_SIZE = 32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    path TEXT PRIMARY KEY,          -- vault-relative, forward slashes
    hash TEXT NOT NULL              -- sha256 of the note's body
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    note_path  TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,    -- position within the note
    title      TEXT NOT NULL,
    headings   TEXT NOT NULL,       -- JSON list
    text       TEXT NOT NULL,
    links      TEXT NOT NULL,       -- JSON list
    embedding  BLOB NOT NULL,       -- float32, length 1
    UNIQUE (note_path, idx)
);
"""


class Embedder(Protocol):
    """Anything that turns texts into vectors. Tests use a fake one."""

    @property
    def model(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Embeds through a local Ollama server (address from `OLLAMA_HOST`, or the
    default http://localhost:11434).

    By default the embedding model runs on the CPU (`num_gpu: 0`), keeping the
    GPU for the answer model: measured, the answer model alone uses
    about 2.9 GB of the ~3 GB budget, and nomic-embed-text on the GPU added
    ~0.3 GB more. On the CPU a question embeds in ~20 ms; re-embedding a whole
    note takes longer but only happens when the note changes.
    """

    def __init__(
        self,
        model: str,
        client: ollama.Client | None = None,
        on_gpu: bool = False,
    ) -> None:
        self._model = model
        # A timeout, so a stuck Ollama raises instead of hanging every question.
        self._client = client or ollama.Client(timeout=120.0)
        self._options = {} if on_gpu else {"num_gpu": 0}

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # truncate=False: Ollama would otherwise silently cut off input longer
        # than the model's context; we'd rather get an error (chunks are sized
        # to fit, see chunking.MAX_CHUNK_CHARS).
        response = self._client.embed(
            model=self._model, input=texts, truncate=False, options=self._options
        )
        return [list(v) for v in response.embeddings]


@dataclass(frozen=True)
class IndexSettings:
    """Everything besides a note's own content that decides what is embedded."""

    model: str
    document_prefix: str
    max_chunk_chars: int = MAX_CHUNK_CHARS
    chunker_version: str = CHUNKER_VERSION
    schema_version: str = SCHEMA_VERSION

    def fingerprint(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class SyncStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: int = 0  # unreadable notes and notes that failed to embed
    chunks_embedded: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.added} added, {self.updated} updated, {self.removed} removed, "
            f"{self.unchanged} unchanged, {self.skipped} skipped "
            f"({self.chunks_embedded} chunks embedded) in {self.seconds:.1f}s"
        )


class IndexStore:
    """The SQLite index file. Use as a context manager so it gets closed:

    with IndexStore(path, IndexSettings("nomic-embed-text", "search_document: ")) as s:
        ...
    """

    def __init__(self, path: Path, settings: IndexSettings | None) -> None:
        """Open (or create) the index at `path`.

        `settings` describe how chunks will be embedded when writing. If the
        index was built with different settings, it is emptied. Pass None to
        open it for searching only, without that check (the retriever checks
        the embedding model instead).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.settings = settings
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA foreign_keys = ON")
        with self._db:
            self._db.executescript(_SCHEMA)
        if settings is not None:
            self._use_settings(settings)

    def __enter__(self) -> IndexStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    # --- metadata ---------------------------------------------------------

    def meta(self, key: str) -> str | None:
        row = self._db.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _use_settings(self, settings: IndexSettings) -> None:
        stored = self.meta("settings")
        if stored == settings.fingerprint():
            return
        if stored is not None:
            log.warning(
                "index settings changed (was %s, now %s); clearing it to rebuild",
                stored,
                settings.fingerprint(),
            )
        with self._db:
            self._db.execute("DELETE FROM chunks")
            self._db.execute("DELETE FROM notes")
            self._db.execute("DELETE FROM meta")
            self._set_meta("settings", settings.fingerprint())
            self._set_meta("embedding_model", settings.model)

    # --- notes and chunks -------------------------------------------------

    def put_note(
        self,
        path: Path,
        note_hash: str,
        chunks: Sequence[Chunk],
        vectors: NDArray[np.float32],
    ) -> None:
        """Replace everything stored for one note, in a single transaction."""
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk is required")
        key = path.as_posix()
        with self._db:
            if len(vectors):
                self._check_dimension(vectors.shape[1])
            self._db.execute("DELETE FROM notes WHERE path = ?", (key,))
            self._db.execute(
                "INSERT INTO notes (path, hash) VALUES (?, ?)", (key, note_hash)
            )
            self._db.executemany(
                "INSERT INTO chunks"
                " (note_path, idx, title, headings, text, links, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        key,
                        c.index,
                        c.title,
                        json.dumps(c.headings, ensure_ascii=False),
                        c.text,
                        json.dumps(c.links, ensure_ascii=False),
                        v.tobytes(),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def delete_note(self, path: str) -> None:
        """Remove a note and its chunks (one transaction; chunks cascade)."""
        with self._db:
            self._db.execute("DELETE FROM notes WHERE path = ?", (path,))

    def note_hashes(self) -> dict[str, str]:
        """Vault-relative path -> stored note hash."""
        rows = self._db.execute("SELECT path, hash FROM notes").fetchall()
        return {str(p): str(h) for p, h in rows}

    def count_chunks(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0])

    def all_chunks(self) -> tuple[list[Chunk], NDArray[np.float32]]:
        """Every stored chunk, and a matrix with one stored vector per row.

        Row i of the matrix belongs to chunk i. With an empty index the matrix
        has shape (0, 0).
        """
        rows = self._db.execute(
            "SELECT note_path, idx, title, headings, text, links, embedding"
            " FROM chunks ORDER BY note_path, idx"
        ).fetchall()
        chunks = [
            Chunk(
                note_path=Path(r[0]),
                title=str(r[2]),
                headings=tuple(json.loads(r[3])),
                index=int(r[1]),
                text=str(r[4]),
                links=tuple(json.loads(r[5])),
            )
            for r in rows
        ]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        vectors = np.stack([np.frombuffer(r[6], dtype=np.float32) for r in rows])
        return chunks, vectors

    def _check_dimension(self, dimension: int) -> None:
        stored = self.meta("dimension")
        if stored is None:
            self._set_meta("dimension", str(dimension))
        elif int(stored) != dimension:
            raise ValueError(
                f"embedding size {dimension} doesn't match the index ({stored})"
            )


def note_hash(note: Note) -> str:
    """Fingerprint of a note's indexed content, to spot changed notes."""
    return hashlib.sha256(note.body.encode("utf-8")).hexdigest()


def embedding_input(chunk: Chunk, prefix: str) -> str:
    """The text sent to the embedding model for one chunk."""
    return f"{prefix}{chunk.label}\n\n{chunk.text}"


def normalize(vectors: Sequence[Sequence[float]]) -> NDArray[np.float32]:
    """Stack vectors into a float32 matrix with each row scaled to length 1."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size == 0:
        return matrix.reshape(0, 0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding model returned a zero vector")
    return matrix / norms


def index_note(
    note: Note,
    store: IndexStore,
    embedder: Embedder,
    batch_size: int = EMBED_BATCH_SIZE,
) -> int:
    """Chunk, embed and store one note, replacing what was stored for it.

    Returns the number of chunks embedded. Nothing is written unless every
    chunk embedded successfully.
    """
    settings = _writable(store, embedder)

    def embed(max_chars: int) -> tuple[list[Chunk], list[list[float]]]:
        chunks = chunk_note(note, max_chars)
        inputs = [embedding_input(c, settings.document_prefix) for c in chunks]
        raw: list[list[float]] = []
        for i in range(0, len(inputs), batch_size):
            raw.extend(embedder.embed(inputs[i : i + batch_size]))
        return chunks, raw

    try:
        chunks, raw = embed(settings.max_chunk_chars)
    except ollama.ResponseError:
        # Dense text (Japanese, tables, URL lists) has more tokens per character
        # and can go over the embedder's context: try once with smaller chunks.
        chunks, raw = embed(settings.max_chunk_chars // 2)
    store.put_note(note.path, note_hash(note), chunks, normalize(raw))
    return len(chunks)


def sync_index(
    vault: Path,
    include: Mapping[str, Sequence[str]],
    store: IndexStore,
    embedder: Embedder,
    batch_size: int = EMBED_BATCH_SIZE,
) -> SyncStats:
    """Bring the index in line with the vault (read only), embedding only what
    changed.

    - New or edited notes that pass the filter are embedded.
    - Notes gone from the vault, or that stopped passing the filter, are removed.
    - A note that can't be read now (broken frontmatter, bad encoding) is logged
      and skipped. If it was indexed before, its old version is kept: a typo
      mid-edit shouldn't make lore vanish, and it's re-indexed once fixed.
    - A note that fails to embed is logged and skipped, keeping any old version.
      Errors reaching Ollama at all, a missing embedding model, and every note
      failing with none indexed are not caught: startup should fail loudly.
    - A vault with no notes at all while the index has some is an error: the
      folder is probably missing for a moment, and the index is left alone.
    """
    _writable(store, embedder)
    if store.path.resolve().is_relative_to(vault.resolve()):
        raise ValueError(f"the index must be outside the vault: {store.path}")
    start = time.perf_counter()

    loaded = load_notes(vault, include)
    stored = store.note_hashes()
    if loaded.found == 0 and stored:
        raise ValueError(f"no notes found in {vault}; keeping the index as it is")
    current = {note.path.as_posix(): note for note in loaded.notes}
    unreadable = {p.as_posix() for p in loaded.errors}

    added = updated = unchanged = removed = chunks = 0
    skipped = len(loaded.errors)
    failure: Exception | None = None
    for key, note in current.items():
        old_hash = stored.get(key)
        if old_hash == note_hash(note):
            unchanged += 1
            continue
        try:
            chunks += index_note(note, store, embedder, batch_size)
        except (ollama.ResponseError, ValueError) as e:
            if isinstance(e, ollama.ResponseError) and e.status_code == 404:
                raise  # the embedding model isn't pulled: every note would fail
            log.warning("skipped note %s: embedding failed: %s", key, e)
            skipped += 1
            failure = e
            continue
        if old_hash is None:
            added += 1
        else:
            updated += 1

    if failure is not None and not (added or updated or unchanged):
        raise failure  # nothing usable is indexed: Ollama is broken, not one note

    for key in stored.keys() - current.keys() - unreadable:
        store.delete_note(key)
        removed += 1

    stats = SyncStats(
        added=added,
        updated=updated,
        removed=removed,
        unchanged=unchanged,
        skipped=skipped,
        chunks_embedded=chunks,
        seconds=time.perf_counter() - start,
    )
    log.info("index sync: %s", stats.summary())
    return stats


def _writable(store: IndexStore, embedder: Embedder) -> IndexSettings:
    if store.settings is None:
        raise ValueError("index was opened for searching only (settings=None)")
    if store.settings.model != embedder.model:
        raise ValueError(
            f"embedder model {embedder.model!r} doesn't match the index "
            f"settings ({store.settings.model!r})"
        )
    return store.settings
