"""Split a note into chunks along its headings, and make wikilinks readable.

A chunk is one heading's section of a note. Each keeps the note's
path and title and the path of headings above it, so answers can cite the note
and the embedder can be given context. `[[wikilinks]]` become plain readable
text and their targets are recorded.

Size: nomic-embed-text has a 2,048-token context (`ollama show nomic-embed-text`),
and Ollama's embed API silently cuts off longer input unless `truncate` is false.
So sections longer than `MAX_CHUNK_CHARS` are split further at paragraph, then
line, then word boundaries. Nothing is cut off.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from loremaster.vault import Note

# About 500 tokens of English (roughly 4 characters per token), leaving room
# under the embedder's 2,048 tokens for denser text and a title/heading prefix,
# and keeping chunks small enough that several fit in the answer model's prompt.
MAX_CHUNK_CHARS = 2000

# Bump this whenever a code change alters what chunk_note produces for the same
# note (headings, splitting, wikilink text). The index stores it and rebuilds
# itself when it changes.
CHUNKER_VERSION = "1"

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
# ![[embed]] or [[target#heading|alias]]; "\|" is the pipe Obsidian uses in tables.
_WIKILINK = re.compile(r"(!?)\[\[([^\[\]]+?)\]\]")
_ALIAS_SPLIT = re.compile(r"\\?\|")


@dataclass(frozen=True)
class Chunk:
    note_path: Path  # relative to the vault
    title: str  # the note's name as Obsidian shows it (file name without .md)
    headings: tuple[str, ...]  # e.g. ("Harbor", "Ferries"); empty before any heading
    index: int  # position of this chunk within its note, from 0
    text: str  # section text with wikilinks made readable
    links: tuple[str, ...]  # wikilink targets in this section, in order, no repeats

    @property
    def label(self) -> str:
        """`Title > Heading > Subheading`, for citations and embedding context."""
        return " > ".join((self.title, *self.headings))


def chunk_note(note: Note, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split `note` into chunks. Sections with no text produce no chunk."""
    chunks: list[Chunk] = []
    for section in find_sections(note.body):
        # Line endings become "\n", whatever the note uses.
        raw = "\n".join(note.body[section.start : section.end].splitlines())
        text, links = clean_wikilinks(raw)
        text = text.strip()
        if not text:
            continue
        for piece in (p.strip() for p in _split_to_size(text, max_chars)):
            chunks.append(
                Chunk(
                    note_path=note.path,
                    title=note.path.stem,
                    headings=section.headings,
                    index=len(chunks),
                    text=piece,
                    links=links,
                )
            )
    return chunks


def clean_wikilinks(text: str) -> tuple[str, tuple[str, ...]]:
    """Replace wikilinks with readable text and return the link targets.

    - `[[Note]]` -> `Note`
    - `[[Note|shown text]]` -> `shown text`
    - `[[Note#Heading]]` -> `Note > Heading`; `[[#Heading]]` -> `Heading`
    - `[[Note#^block]]` -> `Note`
    - `![[embed]]` (images, transcluded notes) is removed from the text; an
      embedded note (no file extension, or `.md`) is still recorded as a link.
    Targets are note names without the `#heading` part, in order, no repeats.
    """
    targets: list[str] = []

    def replace(m: re.Match[str]) -> str:
        embed, inner = m.group(1) == "!", m.group(2)
        parts = _ALIAS_SPLIT.split(inner, maxsplit=1)
        link = parts[0].strip()
        alias = parts[1].strip() if len(parts) > 1 else ""
        target, _, anchor = link.partition("#")
        target = target.strip()
        anchor = anchor.strip()
        is_note = not embed or Path(target).suffix in ("", ".md")
        if target and is_note:
            name = target.removesuffix(".md")
            if name not in targets:
                targets.append(name)
        if embed:
            return ""
        if alias:
            return alias
        if anchor.startswith("^"):
            anchor = ""
        return " > ".join(p for p in (target, anchor) if p)

    return _WIKILINK.sub(replace, text), tuple(targets)


@dataclass(frozen=True)
class Section:
    """One heading's section of a note body, with character positions."""

    headings: tuple[str, ...]  # heading path, as in `Chunk.headings`
    start: int  # where the section's text starts (after its heading line)
    end: int  # where it ends (the next heading line, or the end of the body)


def find_sections(body: str) -> list[Section]:
    """The sections of a note body, in order. Text before the first heading is
    a section with an empty heading path. Lines inside fenced code blocks are
    never treated as headings. `/lore update` uses the positions to edit one
    section in place."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading text) of open headings
    fence: str | None = None  # the opening fence while inside a code block
    start = 0
    pos = 0
    for bare, line in zip(
        body.splitlines(), body.splitlines(keepends=True), strict=True
    ):
        fence_match = _FENCE.match(bare)
        if fence is not None:
            if fence_match and fence_match.group(1).startswith(fence):
                fence = None
        elif fence_match:
            fence = fence_match.group(1)
        elif heading := _HEADING.match(bare):
            sections.append(Section(tuple(h for _, h in stack), start, pos))
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2)))
            start = pos + len(line)
        pos += len(line)
    sections.append(Section(tuple(h for _, h in stack), start, pos))
    return sections


def _split_to_size(text: str, max_chars: int) -> list[str]:
    """Split text into pieces of at most `max_chars`, at the largest boundary
    that works: paragraphs, then lines, then words, then characters."""
    if len(text) <= max_chars:
        return [text]
    for separator in ("\n\n", "\n", " "):
        parts = [p for p in text.split(separator) if p.strip()]
        if len(parts) > 1:
            return _pack(parts, separator, max_chars)
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _pack(parts: list[str], separator: str, max_chars: int) -> list[str]:
    """Join parts back together greedily, keeping each piece within the limit."""
    pieces: list[str] = []
    current = ""
    for part in parts:
        part = part.strip("\n")
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(part) <= max_chars:
            current = part
        else:
            *full, current = _split_to_size(part, max_chars)
            pieces.extend(full)
    if current:
        pieces.append(current)
    return pieces
