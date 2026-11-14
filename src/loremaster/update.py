"""`/lore update`: turn a staff member's lore into a change to the vault.

This is the only module that writes to a vault, and only when the config turns
`/lore update` on. The steps, each a plain function or method so it can be
tested without Discord:

1. **Find the related note**: search the index with the submitted lore.
   If the best match scores at least `update.related_min_score`, propose
   updating that note; otherwise propose a new note. The staff member can
   switch either way.
2. **Propose a change** with the answer model:
   - *Update*: only the section (heading) the best match came from is
     changed; the frontmatter and every other section stay byte for byte as
     they were. The model rewrites that section with the new lore merged in,
     but the rewrite is used only if it kept the original lines (see
     `rewrite_is_safe`: the 4B model sometimes drops paragraphs). Otherwise,
     or if the section is too long for the model's context, the model writes
     text that is added to the end of the section instead.
   - *New note*: the model writes a short note from the lore, starting
     with a `# Title` heading. It goes in the inbox folder, with
     frontmatter from the config or, if none is set, frontmatter
     that matches the include filter.
3. **Preview** happens in the bot: nothing below writes anything.
4. **Apply** after the staff member confirms: the target
   must be a `.md` file inside the vault; an existing note must not have
   changed since the preview; the original is copied to the backup folder;
   the new text goes to a temporary dot-file next to the note
   (ignored by the watcher), which then replaces the note in one step, so the
   note is never half-written. Any failure leaves the note as it was.

The watcher notices the write and re-indexes the note.
"""

import hashlib
import io
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from ruamel.yaml import YAML

from loremaster.answer import ChatModel
from loremaster.chunking import Chunk, Section, find_sections
from loremaster.config import UpdateConfig
from loremaster.retrieve import Hit, Retriever
from loremaster.vault import split_frontmatter

# Sections longer than this (characters) are not rewritten whole: the rewrite
# must fit in the model's reply cap (answer.MAX_ANSWER_TOKENS, about 1,400
# characters), or it comes back cut off. For longer sections the model writes
# an addition instead.
MAX_REWRITE_CHARS = 1000
MAX_TITLE_CHARS = 80
_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|#^\[\]]+')

UPDATE_RULES = """\
You edit one section of a note in a roleplay setting's lore vault.
Merge the staff member's new lore into the section:
- Keep every existing fact unless the new lore changes it.
- Add the new lore where it fits. Don't add facts of your own.
- Keep the section's style and Markdown (lists, bold, [[links]]).
Return only the new section text, without its heading, and nothing else."""

APPEND_RULES = """\
You add the staff member's new lore to the end of one section of a note in a \
roleplay setting's lore vault.
- State only the new lore, in one to three plain sentences (or one list item, \
if the section is a list).
- Don't describe, explain or embellish it, and don't add details of your own.
- Don't repeat anything the section already says.
Return only the text to add, and nothing else."""

NEW_NOTE_RULES = """\
You write a new note for a roleplay setting's lore vault from a staff \
member's lore.
- Start with a level-1 heading naming the topic: "# Title".
- Then write the lore as clear Markdown: a few short paragraphs or a list.
- Use only what the staff member wrote. Don't add facts of your own.
Return only the note, and nothing else."""


class UpdateError(Exception):
    """A change can't be proposed or applied; the message is safe to show staff."""


def replace_section(body: str, section: Section, new_text: str) -> str:
    """`body` with `section`'s text replaced by `new_text`, keeping the blank
    lines around it the way they were (line endings follow the note's)."""
    newline = "\r\n" if "\r\n" in body else "\n"
    old = body[section.start : section.end]
    lead = old[: len(old) - len(old.lstrip())]
    trail = old[len(old.rstrip()) :]
    if not old.strip():  # an empty section: give the new text its own lines
        lead = ""
        trail = newline * 2 if section.end < len(body) else newline
    text = newline.join(new_text.strip().splitlines())
    return body[: section.start] + lead + text + trail + body[section.end :]


# --- checking a rewrite -------------------------------------------------------

# Han, kana, hangul and other CJK blocks: a small model sometimes slips a stray
# character from one of these into English text.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def rewrite_is_safe(old: str, new: str, lore: str) -> bool:
    """Whether a rewritten section kept the original, so it can replace it.

    Seen with the 4B model: rewrites that drop whole paragraphs,
    turn paragraphs into lists, or insert a stray CJK character. So a rewrite
    is used only if (1) nearly every original line is still there word for
    word (one changed line allowed, or one in five for long sections, so a
    fact can be corrected), (2) it isn't much shorter, and (3) it adds no
    CJK characters that weren't in the section or the lore. Otherwise the
    caller adds the new lore at the end of the section instead.
    """
    lines = [line.strip() for line in old.splitlines() if line.strip()]
    kept = {line.strip() for line in new.splitlines()}
    missing = sum(1 for line in lines if line not in kept)
    if missing > max(1, len(lines) // 5):
        return False
    if len(new.strip()) < 0.8 * len(old.strip()):
        return False
    return not has_stray_cjk(new, old + lore)


def merge_addition(old: str, reply: str) -> tuple[str, str]:
    """Add the model's text to a section: (new section text, text added).

    The 4B model often repeats a line of the section with the new fact tacked
    on (e.g. a list item with one more sentence). Such a line extends the
    original line in place instead of duplicating it. Lines the section already
    has word for word are dropped. Everything else goes at the end.
    """
    lines = old.splitlines()
    existing = {line.strip() for line in lines if line.strip()}
    extra: list[str] = []
    added: list[str] = []
    for line in reply.splitlines():
        text = line.strip()
        if not text or text in existing:
            continue
        base = next(
            (
                i
                for i, o in enumerate(lines)
                if len(o.strip()) >= 20 and text.startswith(o.strip())
            ),
            None,
        )
        if base is not None:
            indent = lines[base][: len(lines[base]) - len(lines[base].lstrip())]
            lines[base] = indent + text
        else:
            extra.append(line.rstrip())
        added.append(text)
    section = "\n".join(lines).strip()
    tail = "\n".join(extra).strip()
    return (f"{section}\n\n{tail}" if tail else section), "\n".join(added)


def has_stray_cjk(text: str, source: str) -> bool:
    """Whether `text` has CJK characters that don't appear in `source`."""
    allowed = set(_CJK.findall(source))
    return any(ch not in allowed for ch in _CJK.findall(text))


# --- proposals --------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    """A proposed change, shown to staff before anything is written."""

    kind: Literal["update", "new"]
    path: Path  # vault-relative, e.g. Places/harbor.md
    new_text: str  # the whole file as it will be written
    old_hash: str | None  # sha256 of the file when proposed; None for a new note
    section: str  # what changed, e.g. "harbor > Ferries" or the new note's title
    added: str  # the new or rewritten text, for the preview
    mode: Literal["rewrite", "append", "new"]

    @property
    def title(self) -> str:
        return self.path.stem


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_reply(text: str) -> str:
    """The model's reply without a surrounding ``` fence, if it added one."""
    text = text.strip()
    fenced = re.fullmatch(r"```[\w-]*\n(.*?)\n?```", text, re.S)
    return fenced.group(1).strip() if fenced else text


def safe_title(text: str) -> str:
    """A file-name-safe note title (Obsidian can't use \\ / : * ? " < > | # ^ [ ])."""
    title = _BAD_FILENAME_CHARS.sub(" ", text)
    title = " ".join(title.split()).strip(" .")
    return title[:MAX_TITLE_CHARS].rstrip(" .") or "New lore"


def new_note_parts(reply: str, lore: str) -> tuple[str, str]:
    """(title, body) from the model's new-note reply. The title comes from the
    first `# ` heading, else from the start of the lore; a missing heading is
    added so the note has one."""
    body = strip_reply(reply)
    first = body.splitlines()[0] if body else ""
    if first.startswith("# "):
        return safe_title(first[2:]), body
    title = safe_title(" ".join(lore.split()[:6]))
    return title, f"# {title}\n\n{body}" if body else f"# {title}\n\n{lore.strip()}"


def frontmatter_text(fields: Mapping[str, str | Sequence[str]]) -> str:
    """A `---` frontmatter block for `fields`."""
    yaml = YAML(typ="safe", pure=True)
    yaml.default_flow_style = False
    out = io.StringIO()
    yaml.dump({k: v if isinstance(v, str) else list(v) for k, v in fields.items()}, out)
    return f"---\n{out.getvalue()}---\n"


def default_frontmatter(include: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Frontmatter that passes the include filter: each field's first value."""
    return {field: values[0] for field, values in include.items() if values}


class Updater:
    """Proposes and applies `/lore update` changes for one vault."""

    def __init__(
        self,
        vault: Path,
        retriever: Retriever,
        model: ChatModel,
        settings: UpdateConfig,
        include: Mapping[str, Sequence[str]],
    ) -> None:
        self.vault = vault.resolve()
        self.retriever = retriever
        self.model = model
        self.settings = settings
        self.include = include

    # 1. related note

    def best_match(self, lore: str) -> Hit | None:
        """The best-matching chunk for the lore, however low it scores."""
        hits = self.retriever.search(lore, k=1)
        return hits[0] if hits else None

    def is_related(self, hit: Hit | None) -> bool:
        return hit is not None and hit.score >= self.settings.related_min_score

    # 2. proposals

    def propose(self, lore: str, target: Chunk | None) -> Change:
        """Propose updating `target`'s note (its section), or a new note if None."""
        if not lore.strip():
            raise UpdateError("The lore text is empty.")
        if target is None:
            return self.propose_new(lore)
        return self.propose_update(lore, target)

    def propose_update(self, lore: str, target: Chunk) -> Change:
        path = self._vault_file(target.note_path)
        try:
            data = path.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise UpdateError(f"Couldn't read {target.title}: {e}") from e
        _, body = split_frontmatter(text)
        prefix = text[: len(text) - len(body)]  # frontmatter, kept as it is
        section = next(
            (s for s in find_sections(body) if s.headings == target.headings), None
        )
        if section is None:
            raise UpdateError(
                f"The section {target.label!r} is no longer in {target.title}; "
                "try again in a moment."
            )
        old = body[section.start : section.end].strip()
        label = target.label
        mode: Literal["rewrite", "append"] = "append"
        if len(old) <= MAX_REWRITE_CHARS:
            prompt = (
                f"Note: {target.title}\nSection: {label}\n\n"
                f"Current section text:\n<<<\n{old}\n>>>\n\n"
                f"New lore from staff:\n<<<\n{lore.strip()}\n>>>"
            )
            reply = strip_reply(self._chat(UPDATE_RULES, prompt))
            # The model sometimes repeats the section's heading, which would
            # double it; a section's own text never starts with a heading.
            reply = re.sub(r"\A#{1,6}[ \t].*\n*", "", reply)
            if not reply:
                raise UpdateError("The model returned an empty section.")
            if rewrite_is_safe(old, reply, lore):
                mode = "rewrite"
                new_section = reply
                added = reply
        if mode == "append":
            tail = old[-1500:]
            prompt = (
                f"Note: {target.title}\nSection: {label}\n\n"
                f"End of the current section:\n<<<\n{tail}\n>>>\n\n"
                f"New lore from staff:\n<<<\n{lore.strip()}\n>>>"
            )
            reply = strip_reply(self._chat(APPEND_RULES, prompt))
            if not reply:
                raise UpdateError("The model returned nothing to add.")
            if has_stray_cjk(reply, old + lore):
                raise UpdateError("The model's text came out garbled; try again.")
            new_section, added = merge_addition(old, reply)
            if not added:
                raise UpdateError("The model only repeated what the note already says.")
        new_body = replace_section(body, section, new_section)
        return Change(
            kind="update",
            path=target.note_path,
            new_text=prefix + new_body,
            old_hash=file_hash(data),
            section=label,
            added=added,
            mode=mode,
        )

    def propose_new(self, lore: str) -> Change:
        reply = self._chat(
            NEW_NOTE_RULES, f"Lore from staff:\n<<<\n{lore.strip()}\n>>>"
        )
        title, body = new_note_parts(reply, lore)
        if has_stray_cjk(body, lore):
            raise UpdateError("The model's text came out garbled; try again.")
        fields = self.settings.new_note_frontmatter or default_frontmatter(self.include)
        path = self._free_path(title)
        newline_body = body.strip() + "\n"
        return Change(
            kind="new",
            path=path,
            new_text=frontmatter_text(fields) + "\n" + newline_body,
            old_hash=None,
            section=path.stem,
            added=newline_body.strip(),
            mode="new",
        )

    # 4. apply

    def apply(self, change: Change, now: datetime | None = None) -> Path:
        """Write a confirmed change. Returns the file written. Raises
        `UpdateError` (and leaves the vault unchanged) if anything is wrong."""
        path = self._vault_file(change.path)
        if change.kind == "update":
            try:
                current = path.read_bytes()
            except OSError as e:
                raise UpdateError(f"Couldn't read {change.title}: {e}") from e
            if file_hash(current) != change.old_hash:
                raise UpdateError(
                    f"{change.title} changed since the preview; nothing was "
                    "written. Run the command again."
                )
            self._backup(change.path, current, now or datetime.now())
        elif path.exists():
            raise UpdateError(
                f"{change.path.as_posix()} already exists; nothing was written."
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, change.new_text.encode("utf-8"))
        except OSError as e:
            raise UpdateError(f"Couldn't write {change.title}: {e}") from e
        return path

    # helpers

    def _chat(self, rules: str, prompt: str) -> str:
        return self.model.chat(
            [{"role": "system", "content": rules}, {"role": "user", "content": prompt}]
        )

    def _vault_file(self, rel: Path) -> Path:
        """The absolute path of a vault-relative note, refusing anything that
        isn't a `.md` file inside the vault."""
        if rel.is_absolute() or rel.suffix.lower() != ".md":
            raise UpdateError(f"Refusing to write {rel.as_posix()}: not a note path.")
        path = (self.vault / rel).resolve()
        if not path.is_relative_to(self.vault) or path == self.vault:
            raise UpdateError(f"Refusing to write {rel.as_posix()}: outside the vault.")
        return path

    def _free_path(self, title: str) -> Path:
        """`inbox/Title.md`, numbered if a note with that name exists anywhere in
        the vault (Obsidian links notes by file name, so names stay unique)."""
        taken = {p.stem.lower() for p in self.vault.rglob("*.md")}
        name = title
        n = 2
        while name.lower() in taken:
            name = f"{title} {n}"
            n += 1
        return self.settings.inbox_folder / f"{name}.md"

    def _backup(self, rel: Path, data: bytes, now: datetime) -> Path:
        target = self.settings.backup_path / now.strftime("%Y%m%d-%H%M%S") / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as e:
            raise UpdateError(
                f"Couldn't back up {rel.stem}; nothing was written: {e}"
            ) from e
        return target


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temporary dot-file in the same folder, then replace `path`
    in one step. The dot-file is ignored by the watcher and removed on error."""
    fd, tmp = tempfile.mkstemp(prefix=".loremaster-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
