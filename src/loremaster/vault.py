"""Walk a vault, parse each note's frontmatter, and keep the notes the config allows.

The vault is only read here, never written. A note whose frontmatter
can't be parsed is logged and skipped; it never stops the rest of the load
.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

log = logging.getLogger(__name__)

# Frontmatter after YAML parsing. Values can be anything YAML produces
# (strings, numbers, lists, nested mappings), so they are typed as `object`
# and narrowed where used.
Frontmatter = dict[str, object]


class FrontmatterError(Exception):
    """A note's frontmatter block exists but can't be read as a YAML mapping."""


@dataclass(frozen=True)
class Note:
    """One note the filter kept."""

    path: Path  # relative to the vault root, e.g. Path("Places/harbor.md")
    frontmatter: Frontmatter
    body: str  # the Markdown after the frontmatter block


@dataclass
class LoadResult:
    """The kept notes, plus what was skipped and why (for logs and smoke runs)."""

    notes: list[Note] = field(default_factory=list)
    found: int = 0
    skipped_no_frontmatter: int = 0
    skipped_not_matching: int = 0
    errors: list[Path] = field(default_factory=list)  # bad frontmatter/unreadable


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Split a note into (frontmatter YAML text, body).

    Frontmatter is a block that starts on the note's first line with `---` and
    ends at the next line that is exactly `---` (as in Obsidian). Returns
    `(None, text)` when the note has no frontmatter block. Raises
    `FrontmatterError` when the block is opened but never closed.
    """
    text = text.removeprefix("﻿")  # a byte-order mark some editors add
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return None, text
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            return "".join(lines[1:i]), "".join(lines[i + 1 :])
    raise FrontmatterError("frontmatter block is not closed with '---'")


def parse_frontmatter(text: str) -> tuple[Frontmatter | None, str]:
    """Return (frontmatter mapping, body). The mapping is None if there is none.

    An empty block (`---` followed directly by `---`) counts as no frontmatter.
    """
    raw, body = split_frontmatter(text)
    if raw is None:
        return None, body
    try:
        data: object = YAML(typ="safe", pure=True).load(raw)
    except YAMLError as e:
        raise FrontmatterError(f"invalid YAML: {e}") from e
    if data is None:
        return None, body
    if not isinstance(data, dict):
        raise FrontmatterError("frontmatter is not a mapping of keys to values")
    return {str(k): v for k, v in data.items()}, body


def matches(frontmatter: Frontmatter, include: Mapping[str, Sequence[str]]) -> bool:
    """True if the note passes the include filter.

    Rules: every listed field must match (AND). A field matches when the
    note's value is one of the allowed values; when the note's value is a list,
    when any item is allowed. Compared as strings, case-sensitive. A missing
    field means no match.
    """
    for key, allowed in include.items():
        if key not in frontmatter:
            return False
        value = frontmatter[key]
        values = value if isinstance(value, list) else [value]
        if not any(_as_text(v) in allowed for v in values):
            return False
    return True


def load_notes(vault: Path, include: Mapping[str, Sequence[str]]) -> LoadResult:
    """Read every `.md` note in `vault` and keep those matching `include`.

    Folders and files whose name starts with "." (such as `.obsidian`, `.git`,
    `.trash`) are ignored.
    """
    result = LoadResult()
    for file in sorted(vault.rglob("*.md")):
        rel = file.relative_to(vault)
        if any(part.startswith(".") for part in rel.parts) or not file.is_file():
            continue
        result.found += 1
        try:
            frontmatter, body = parse_frontmatter(file.read_text(encoding="utf-8"))
        except (FrontmatterError, OSError, UnicodeDecodeError) as e:
            log.warning("skipped note %s: %s", rel.as_posix(), e)
            result.errors.append(rel)
            continue
        if frontmatter is None:
            result.skipped_no_frontmatter += 1
        elif not matches(frontmatter, include):
            result.skipped_not_matching += 1
        else:
            result.notes.append(Note(path=rel, frontmatter=frontmatter, body=body))
    return result


def _as_text(value: object) -> str:
    # YAML turns `true` into a bool; compare it as the text "true" like Obsidian shows.
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
