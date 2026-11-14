"""Tests for `/lore update`.

Everything writes to temporary made-up vaults (`tmp_path`), never a real one.
A fake chat model returns fixed replies; a word-count fake embedder does search.
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest

from loremaster.answer import MAX_ANSWER_TOKENS, Answerer, Message
from loremaster.bot import (
    PREVIEW_TIMEOUT_S,
    STAFF_ONLY,
    UPDATE_OFF,
    LoreService,
    Proposal,
    UpdateView,
    check_vault_writable,
    format_applied,
    format_preview,
    make_lore_group,
)
from loremaster.chunking import Chunk, find_sections
from loremaster.config import Config, ConfigError, UpdateConfig, load_config
from loremaster.index import IndexStore
from loremaster.retrieve import Hit, Retriever
from loremaster.update import (
    MAX_REWRITE_CHARS,
    Change,
    UpdateError,
    Updater,
    default_frontmatter,
    frontmatter_text,
    merge_addition,
    new_note_parts,
    replace_section,
    rewrite_is_safe,
    safe_title,
    strip_reply,
)
from tests.test_answer import FakeChat
from tests.test_retrieve import NOTES, WordsEmbedder, build

INCLUDE = {"status": ["approved"]}
NOW = datetime(2026, 11, 12, 9, 30, 0)

HARBOR = (
    "---\nstatus: approved\ntags: [port]\n---\n"
    "Intro line about the harbor.\n\n"
    "# Harbor\nThe harbor is busy.\n\n"
    "## Ferries\nFerries leave at dawn.\n\n"
    "```\n# not a heading\n```\n\n"
    "## Market\nFish are sold here.\n"
)


def chunk(path: str, headings: tuple[str, ...]) -> Chunk:
    return Chunk(Path(path), Path(path).stem, headings, 0, "x", ())


# --- sections ---------------------------------------------------------------


def body_of(text: str) -> str:
    return text.split("---\n", 2)[2]


def test_find_sections_paths_and_positions() -> None:
    body = body_of(HARBOR)
    sections = find_sections(body)
    assert [s.headings for s in sections] == [
        (),
        ("Harbor",),
        ("Harbor", "Ferries"),
        ("Harbor", "Market"),
    ]
    ferries = sections[2]
    assert body[ferries.start : ferries.end].startswith("Ferries leave at dawn.")
    assert "# not a heading" in body[ferries.start : ferries.end]  # fence kept


def test_replace_section_changes_only_that_section() -> None:
    body = body_of(HARBOR)
    ferries = find_sections(body)[2]
    new = replace_section(body, ferries, "Ferries leave at dawn and dusk.")
    assert "Ferries leave at dawn and dusk.\n\n## Market\nFish are sold here.\n" in new
    assert new.startswith(
        "Intro line about the harbor.\n\n# Harbor\nThe harbor is busy."
    )
    assert "# not a heading" not in new  # it was part of the replaced section


def test_replace_section_keeps_crlf() -> None:
    body = "# A\r\nold\r\n\r\n# B\r\nkeep\r\n"
    a = find_sections(body)[1]
    assert replace_section(body, a, "new one\nnew two") == (
        "# A\r\nnew one\r\nnew two\r\n\r\n# B\r\nkeep\r\n"
    )


def test_replace_empty_section() -> None:
    body = "# A\n# B\nkeep\n"
    a = find_sections(body)[1]
    assert replace_section(body, a, "filled") == "# A\nfilled\n\n# B\nkeep\n"


# --- small helpers ----------------------------------------------------------


def test_strip_reply_removes_fences() -> None:
    assert strip_reply("```markdown\nHello\n```") == "Hello"
    assert strip_reply("  plain  ") == "plain"


def test_safe_title() -> None:
    assert safe_title('The "Moon" / Sea: part #2?') == "The Moon Sea part 2"
    assert safe_title("...") == "New lore"
    assert len(safe_title("x" * 200)) == 80


def test_new_note_parts() -> None:
    assert new_note_parts("# Glass Bridge\nIt glows.", "lore") == (
        "Glass Bridge",
        "# Glass Bridge\nIt glows.",
    )
    title, body = new_note_parts(
        "It glows at night.", "The glass bridge glows at night"
    )
    assert title == "The glass bridge glows at night"
    assert body.startswith("# The glass bridge glows at night\n\nIt glows")


def test_frontmatter_text_and_default() -> None:
    assert frontmatter_text({"status": "draft", "tags": ["a", "b"]}) == (
        "---\nstatus: draft\ntags:\n- a\n- b\n---\n"
    )
    assert default_frontmatter(
        {"scope": ["setting", "rules"], "status": ["approved"]}
    ) == {
        "scope": "setting",
        "status": "approved",
    }


# --- proposals --------------------------------------------------------------


def make_updater(
    tmp_path: Path,
    chat: FakeChat,
    settings: UpdateConfig | None = None,
    notes: dict[str, str] | None = None,
) -> Updater:
    embedder = WordsEmbedder()
    db = build(tmp_path, notes or NOTES, embedder)
    retriever = Retriever(IndexStore(db, None), embedder, "")
    settings = settings or UpdateConfig(
        enabled=True, backup_path=tmp_path / "backups", related_min_score=0.5
    )
    return Updater(tmp_path / "vault", retriever, chat, settings, INCLUDE)


def write_harbor(updater: Updater) -> Path:
    path = updater.vault / "harbor.md"
    path.write_bytes(HARBOR.encode())
    return path


# The submitted lore is searched against the index.
def test_best_match_and_related(tmp_path: Path) -> None:
    updater = make_updater(tmp_path, FakeChat())
    best = updater.best_match("The bakery now also sells honey cakes on Sundays.")
    assert best is not None and best.chunk.note_path == Path("bakery.md")
    assert updater.is_related(Hit(best.chunk, 0.5))
    assert not updater.is_related(Hit(best.chunk, 0.49))
    assert not updater.is_related(None)


# The related note's section is rewritten; frontmatter and the rest stay.
def test_propose_update_rewrites_one_section(tmp_path: Path) -> None:
    chat = FakeChat("Ferries leave at dawn and at dusk.\n\n```\n# not a heading\n```")
    updater = make_updater(tmp_path, chat)
    write_harbor(updater)
    change = updater.propose(
        "Ferries also leave at dusk.", chunk("harbor.md", ("Harbor", "Ferries"))
    )
    assert change.kind == "update" and change.mode == "rewrite"
    assert change.section == "harbor > Harbor > Ferries"
    assert change.new_text == HARBOR.replace(
        "Ferries leave at dawn.\n",
        "Ferries leave at dawn and at dusk.\n",
    )
    assert change.new_text.startswith("---\nstatus: approved\ntags: [port]\n---\n")
    prompt = chat.prompts[0][1]["content"]
    assert (
        "Ferries leave at dawn." in prompt and "Ferries also leave at dusk." in prompt
    )
    assert "Fish are sold here." not in prompt  # only the one section is sent


def test_rewrite_that_repeats_the_heading_does_not_double_it(tmp_path: Path) -> None:
    chat = FakeChat(
        "## Ferries\nFerries leave at dawn and at dusk.\n\n```\n# not a heading\n```"
    )
    updater = make_updater(tmp_path, chat)
    write_harbor(updater)
    change = updater.propose(
        "Ferries also leave at dusk.", chunk("harbor.md", ("Harbor", "Ferries"))
    )
    assert change.mode == "rewrite"
    assert change.new_text.count("## Ferries") == 1


def test_a_rewrite_fits_in_the_reply_cap() -> None:
    # At a cautious 3 characters per token; a cut-off rewrite can pass as safe.
    assert MAX_REWRITE_CHARS <= 3 * MAX_ANSWER_TOKENS


def test_long_section_gets_an_addition(tmp_path: Path) -> None:
    chat = FakeChat("A night ferry runs on holidays.")
    updater = make_updater(tmp_path, chat)
    long = ("Ferries leave at dawn. " * (MAX_REWRITE_CHARS // 20)).strip()
    (updater.vault / "harbor.md").write_text(f"# Ferries\n{long}\n\n# Market\nFish.\n")
    change = updater.propose(
        "A night ferry runs on holidays.", chunk("harbor.md", ("Ferries",))
    )
    assert change.mode == "append"
    assert change.new_text.endswith(
        f"{long.strip()}\n\nA night ferry runs on holidays.\n\n# Market\nFish.\n"
    )


def test_missing_section_is_an_error(tmp_path: Path) -> None:
    updater = make_updater(tmp_path, FakeChat())
    write_harbor(updater)
    with pytest.raises(UpdateError, match="no longer in harbor"):
        updater.propose("x", chunk("harbor.md", ("Harbor", "Gone")))


def test_empty_model_reply_is_an_error(tmp_path: Path) -> None:
    updater = make_updater(tmp_path, FakeChat("  "))
    write_harbor(updater)
    with pytest.raises(UpdateError, match="empty"):
        updater.propose("x", chunk("harbor.md", ("Harbor", "Ferries")))


def test_empty_lore_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(UpdateError, match="empty"):
        make_updater(tmp_path, FakeChat()).propose("  ", None)


# A new note in the inbox, config frontmatter.
def test_propose_new_note(tmp_path: Path) -> None:
    chat = FakeChat("```\n# Glass Bridge\nThe glass bridge glows at night.\n```")
    settings = UpdateConfig(
        enabled=True,
        backup_path=tmp_path / "b",
        inbox_folder=Path("Inbox"),
        new_note_frontmatter={"status": "draft"},
    )
    change = make_updater(tmp_path, chat, settings).propose(
        "A glass bridge glows.", None
    )
    assert change.kind == "new" and change.old_hash is None
    assert change.path == Path("Inbox/Glass Bridge.md")
    assert change.new_text == (
        "---\nstatus: draft\n---\n\n# Glass Bridge\nThe glass bridge glows at night.\n"
    )


def test_garbled_new_note_is_an_error(tmp_path: Path) -> None:
    chat = FakeChat("# Glass Bridge\nThe glass们 bridge glows.")
    with pytest.raises(UpdateError, match="garbled"):
        make_updater(tmp_path, chat).propose("A glass bridge glows.", None)


def test_new_note_defaults_to_include_frontmatter(tmp_path: Path) -> None:
    change = make_updater(tmp_path, FakeChat("# Glass Bridge\nIt glows.")).propose(
        "x", None
    )
    assert change.new_text.startswith("---\nstatus: approved\n---\n")


def test_new_note_name_stays_unique_in_vault(tmp_path: Path) -> None:
    change = make_updater(tmp_path, FakeChat("# Bakery\nMore bread.")).propose(
        "x", None
    )
    assert change.path == Path("Inbox/Bakery 2.md")  # bakery.md already exists


# --- applying ---------------------------------------------------------------


def update_change(updater: Updater) -> Change:
    write_harbor(updater)
    return updater.propose("x", chunk("harbor.md", ("Harbor", "Ferries")))


# The original is backed up outside the vault before the edit.
def test_apply_update_backs_up_and_writes(tmp_path: Path) -> None:
    updater = make_updater(tmp_path, FakeChat("New ferry text."))
    change = update_change(updater)
    path = updater.apply(change, NOW)
    assert path.read_text() == change.new_text
    backup = tmp_path / "backups" / "20261112-093000" / "harbor.md"
    assert backup.read_bytes() == HARBOR.encode()
    assert not any(p.name.startswith(".loremaster-") for p in updater.vault.iterdir())


# A note changed since the preview is not overwritten.
def test_apply_refuses_if_note_changed(tmp_path: Path) -> None:
    updater = make_updater(tmp_path, FakeChat("New ferry text."))
    change = update_change(updater)
    (updater.vault / "harbor.md").write_text(HARBOR + "\nEdited meanwhile.\n")
    with pytest.raises(UpdateError, match="changed since the preview"):
        updater.apply(change, NOW)
    assert (updater.vault / "harbor.md").read_text().endswith("Edited meanwhile.\n")
    assert not (tmp_path / "backups").exists()


# When writing fails, the existing note is unchanged.
def test_failed_write_leaves_note_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    updater = make_updater(tmp_path, FakeChat("New ferry text."))
    change = update_change(updater)

    def broken_replace(_src: object, _dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("loremaster.update.os.replace", broken_replace)
    with pytest.raises(UpdateError, match="Couldn't write harbor"):
        updater.apply(change, NOW)
    assert (updater.vault / "harbor.md").read_bytes() == HARBOR.encode()
    assert not any(p.name.startswith(".loremaster-") for p in updater.vault.iterdir())


# Only .md notes inside the vault.
@pytest.mark.parametrize(
    "rel", ["../outside.md", "notes.txt", "/etc/passwd.md", "a/../../x.md"]
)
def test_refuses_paths_outside_vault_or_not_notes(tmp_path: Path, rel: str) -> None:
    updater = make_updater(tmp_path, FakeChat())
    change = Change("new", Path(rel), "text", None, "x", "text", "new")
    with pytest.raises(UpdateError, match="Refusing to write"):
        updater.apply(change, NOW)
    assert not (tmp_path / "outside.md").exists()


def test_apply_new_note_creates_inbox(tmp_path: Path) -> None:
    updater = make_updater(tmp_path, FakeChat("# Glass Bridge\nIt glows."))
    change = updater.propose("x", None)
    path = updater.apply(change, NOW)
    assert path == updater.vault / "Inbox" / "Glass Bridge.md"
    assert path.read_text() == change.new_text


def test_apply_new_note_never_overwrites(tmp_path: Path) -> None:
    updater = make_updater(tmp_path, FakeChat("# Glass Bridge\nIt glows."))
    change = updater.propose("x", None)
    (updater.vault / "Inbox").mkdir()
    (updater.vault / "Inbox" / "Glass Bridge.md").write_text("someone else's")
    with pytest.raises(UpdateError, match="already exists"):
        updater.apply(change, NOW)
    assert (updater.vault / "Inbox" / "Glass Bridge.md").read_text() == "someone else's"


# --- config -----------------------------------------------------------------


def write_config(tmp_path: Path, vault: Path, extra: str) -> Path:
    path = tmp_path / "lorebot.yaml"
    path.write_text(
        f"vault_path: {vault.as_posix()}\ninclude:\n  status: [approved]\n{extra}",
        encoding="utf-8",
    )
    return path


def test_update_is_off_by_default(tmp_path: Path) -> None:
    (tmp_path / "v").mkdir()
    config = load_config(write_config(tmp_path, tmp_path / "v", ""))
    assert not config.update.enabled
    assert config.update.backup_path == (Path.cwd() / "data" / "backups").resolve()


def test_update_settings_load(tmp_path: Path) -> None:
    (tmp_path / "v").mkdir()
    config = load_config(
        write_config(
            tmp_path,
            tmp_path / "v",
            "update:\n  enabled: true\n  related_min_score: 0.8\n"
            "  inbox_folder: New/Lore\n  backup_path: bk\n"
            "  new_note_frontmatter:\n    status: draft\n    tags: [new]\n",
        )
    )
    u = config.update
    assert u.enabled and u.related_min_score == 0.8
    assert u.inbox_folder == Path("New/Lore")
    assert u.backup_path == (tmp_path / "bk").resolve()
    assert u.new_note_frontmatter == {"status": "draft", "tags": ["new"]}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            "update:\n  backup_path: v/bk\n",
            "update.backup_path: must be outside the vault",
        ),
        (
            "update:\n  inbox_folder: ../out\n",
            "update.inbox_folder: must be a folder inside",
        ),
        ("update:\n  inbox_folder: .hidden\n", "hidden"),
        ("update:\n  related_min_score: 2\n", "update.related_min_score"),
    ],
)
def test_bad_update_settings(tmp_path: Path, extra: str, message: str) -> None:
    (tmp_path / "v").mkdir()
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, tmp_path / "v", extra))


# Decision 1: with update on, a read-only vault stops startup.
def test_read_only_vault_with_update_on_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "v").mkdir()
    config = load_config(
        write_config(tmp_path, tmp_path / "v", "update:\n  enabled: true\n")
    )
    check_vault_writable(config)  # writable: fine
    monkeypatch.setattr("loremaster.bot.os.access", lambda _p, _m: False)
    with pytest.raises(ConfigError, match="VAULT_MODE=rw"):
        check_vault_writable(config)
    off = config.model_copy(update={"update": UpdateConfig()})
    check_vault_writable(off)  # update off: read-only is fine


# --- the bot ----------------------------------------------------------------


def service_for(
    tmp_path: Path, chat: FakeChat, staff: str = "[7]", enabled: bool = True
) -> LoreService:
    updater = make_updater(tmp_path, chat)
    (tmp_path / "lorebot.yaml").write_text(
        f"vault_path: {updater.vault.as_posix()}\ninclude:\n  status: [approved]\n"
        f"discord:\n  staff_role_ids: {staff}\n",
        encoding="utf-8",
    )
    config: Config = load_config(tmp_path / "lorebot.yaml")
    answerer = Answerer(updater.retriever, chat, "", min_score=-1)
    return LoreService(config, answerer, updater=updater if enabled else None)


# Only members with a staff role; none configured = nobody.
def test_admit_update(tmp_path: Path) -> None:
    service = service_for(tmp_path, FakeChat())
    assert service.admit_update(guild_id=1, role_ids=[3, 7]) is None
    refusal = service.admit_update(guild_id=1, role_ids=[3])
    assert refusal is not None and refusal.message == STAFF_ONLY
    assert service.admit_update(guild_id=None, role_ids=[7]) is not None


def test_no_staff_roles_means_nobody_can_update(tmp_path: Path) -> None:
    service = service_for(tmp_path, FakeChat(), staff="[]")
    assert service.admit_update(guild_id=1, role_ids=[7]) is not None


def test_update_off_refuses(tmp_path: Path) -> None:
    service = service_for(tmp_path, FakeChat(), enabled=False)
    refusal = service.admit_update(guild_id=1, role_ids=[7])
    assert refusal is not None and refusal.message == UPDATE_OFF


# With update off, the command isn't offered.
def test_command_offered_only_when_on(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    on = make_lore_group(service_for(tmp_path / "a", FakeChat()))
    off = make_lore_group(service_for(tmp_path / "b", FakeChat(), enabled=False))
    assert on.get_command("update") is not None
    assert off.get_command("update") is None
    assert off.get_command("ask") is not None


def test_propose_related_lore_updates_the_best_note(tmp_path: Path) -> None:
    chat = FakeChat("The bakery sells rye bread, honey cakes and pretzels.")
    service = service_for(tmp_path, chat)
    result = asyncio.run(
        service.propose(
            "The bakery sells rye bread, honey cakes and pretzels every morning."
        )
    )
    assert isinstance(result, Proposal)
    assert result.change.kind == "update"
    assert result.change.path == Path("bakery.md")
    assert result.switch_label == "Make a new note instead"


def test_propose_unrelated_lore_makes_a_new_note(tmp_path: Path) -> None:
    service = service_for(tmp_path, FakeChat("# Glass Bridge\nIt glows."))
    assert service.updater is not None
    service.updater.settings = service.updater.settings.model_copy(
        update={"related_min_score": 0.99}
    )
    result = asyncio.run(service.propose("A glass bridge glows over the canyon."))
    assert isinstance(result, Proposal) and result.change.kind == "new"
    assert result.switch_label is not None and result.switch_label.startswith("Update ")


# Decision 2: the staff member can switch either way.
def test_switch_both_ways(tmp_path: Path) -> None:
    chat = FakeChat("# Pretzels\nThe bakery sells pretzels.")
    service = service_for(tmp_path, chat)
    first = asyncio.run(
        service.propose(
            "The bakery sells rye bread, honey cakes and pretzels every morning."
        )
    )
    assert isinstance(first, Proposal) and first.change.kind == "update"
    second = asyncio.run(service.propose(first.lore, switch_from=first))
    assert isinstance(second, Proposal) and second.change.kind == "new"
    third = asyncio.run(service.propose(first.lore, switch_from=second))
    assert isinstance(third, Proposal) and third.change.kind == "update"
    assert third.change.path == Path("bakery.md")


def test_propose_errors_become_messages(tmp_path: Path) -> None:
    service = service_for(tmp_path, FakeChat(""))
    result = asyncio.run(
        service.propose(
            "The bakery sells rye bread, honey cakes and pretzels every morning."
        )
    )
    assert result == "The model returned an empty section."


# Nothing is written until apply (Confirm); the reply names the note.
def test_nothing_written_until_apply(tmp_path: Path) -> None:
    service = service_for(tmp_path, FakeChat("Rye bread and pretzels."))
    assert service.updater is not None
    vault = service.updater.vault
    before = {p: p.read_bytes() for p in vault.rglob("*.md")}
    proposal = asyncio.run(
        service.propose(
            "The bakery sells rye bread, honey cakes and pretzels every morning."
        )
    )
    assert isinstance(proposal, Proposal)
    assert {p: p.read_bytes() for p in vault.rglob("*.md")} == before
    reply = asyncio.run(service.apply(proposal))
    assert reply.startswith("Updated **bakery**")
    assert "Rye bread and pretzels." in (vault / "bakery.md").read_text()


def test_apply_errors_become_messages(tmp_path: Path) -> None:
    service = service_for(tmp_path, FakeChat("Rye bread and pretzels."))
    proposal = asyncio.run(
        service.propose(
            "The bakery sells rye bread, honey cakes and pretzels every morning."
        )
    )
    assert isinstance(proposal, Proposal) and service.updater is not None
    (service.updater.vault / "bakery.md").write_text("changed meanwhile")
    reply = asyncio.run(service.apply(proposal))
    assert reply.startswith("Nothing was written:")


def test_preview_text_and_cut() -> None:
    change = Change(
        "update",
        Path("harbor.md"),
        "",
        "h",
        "harbor > Ferries",
        "Line one\n\nLine two",
        "rewrite",
    )
    text, cut = format_preview(Proposal("lore", change, None))
    assert not cut
    assert "**Preview: update harbor** › harbor > Ferries" in text
    assert "> Line one\n>\n> Line two" in text
    assert "Nothing is written until you confirm." in text
    long = Change("new", Path("Inbox/X.md"), "", None, "X", "word " * 1000, "new")
    text, cut = format_preview(Proposal("lore", long, None))
    assert cut and len(text) <= 2000
    assert "attached file" in text


def test_applied_messages() -> None:
    upd = Change(
        "update", Path("harbor.md"), "", "h", "harbor > Ferries", "t", "append"
    )
    assert "added to the section harbor > Ferries" in format_applied(
        upd, Path("harbor.md")
    )
    new = Change("new", Path("Inbox/X.md"), "", None, "X", "t", "new")
    assert "`Inbox/X.md`" in format_applied(new, Path("Inbox/X.md"))


# --- rewrite safety (seen with the real 4B model) ----------------------------


class ScriptedChat(FakeChat):
    """Answers with each reply in turn."""

    def __init__(self, *replies: str) -> None:
        super().__init__()
        self.replies = list(replies)

    def chat(self, messages: Sequence[Message]) -> str:
        self.prompts.append(list(messages))
        return self.replies.pop(0)


OLD = "Spring is warm.\n\nSummer is hot.\n\nAutumn is cool.\n\nWinter is cold."


def test_rewrite_that_keeps_lines_is_safe() -> None:
    new = OLD.replace("Spring is warm.", "Spring is warm, with light snow far out.")
    assert rewrite_is_safe(OLD, new, "light snow in spring")  # one line changed


def test_rewrite_that_drops_lines_is_not_safe() -> None:
    assert not rewrite_is_safe(OLD, "Spring is warm, with light snow far out.", "snow")


def test_rewrite_that_reformats_is_not_safe() -> None:
    bullets = "\n".join(f"- {line}" for line in OLD.split("\n\n"))
    assert not rewrite_is_safe(OLD, bullets, "x")


def test_rewrite_with_stray_cjk_is_not_safe() -> None:
    new = OLD.replace("Summer is hot.", "Summer们 is hot.")
    assert not rewrite_is_safe(OLD, new, "x")
    assert rewrite_is_safe(
        "Spring at 浅間.", "Spring at 浅間 is warm.", "x"
    )  # already there


def test_unsafe_rewrite_falls_back_to_adding(tmp_path: Path) -> None:
    chat = ScriptedChat("Spring has light snow.", "Light snow also falls in spring.")
    updater = make_updater(tmp_path, chat)
    (updater.vault / "climate.md").write_text(f"# Climate\n{OLD}\n\n# Other\nKeep.\n")
    change = updater.propose("Light snow in spring.", chunk("climate.md", ("Climate",)))
    assert change.mode == "append"
    assert change.added == "Light snow also falls in spring."
    assert change.new_text == (
        f"# Climate\n{OLD}\n\nLight snow also falls in spring.\n\n# Other\nKeep.\n"
    )
    assert len(chat.prompts) == 2


def test_garbled_addition_is_an_error(tmp_path: Path) -> None:
    chat = ScriptedChat("dropped everything", "Snow们 falls.")
    updater = make_updater(tmp_path, chat)
    (updater.vault / "climate.md").write_text(f"# Climate\n{OLD}\n")
    with pytest.raises(UpdateError, match="garbled"):
        updater.propose("Light snow in spring.", chunk("climate.md", ("Climate",)))


def test_merge_addition_extends_a_repeated_line_in_place() -> None:
    old = "- **Spring:** Blossoms bloom here.\n- **Summer:** Hot and wet."
    reply = "- **Spring:** Blossoms bloom here. Light snow falls far out."
    section, added = merge_addition(old, reply)
    assert section == (
        "- **Spring:** Blossoms bloom here. Light snow falls far out.\n"
        "- **Summer:** Hot and wet."
    )
    assert added == "- **Spring:** Blossoms bloom here. Light snow falls far out."


def test_merge_addition_appends_new_lines_and_drops_repeats() -> None:
    old = "The city has many parks and large fields."
    reply = "The city has many parks and large fields.\nA puppet library lends books."
    section, added = merge_addition(old, reply)
    assert section == old + "\n\nA puppet library lends books."
    assert added == "A puppet library lends books."
    assert merge_addition(old, old) == (old, "")


def test_preview_buttons(tmp_path: Path) -> None:
    service = service_for(tmp_path, FakeChat("Rye bread and pretzels."))
    proposal = asyncio.run(
        service.propose(
            "The bakery sells rye bread, honey cakes and pretzels every morning."
        )
    )
    assert isinstance(proposal, Proposal)

    async def build_view() -> list[str]:
        view = UpdateView(service, proposal, owner_id=42)
        assert view.timeout == PREVIEW_TIMEOUT_S == 600  # 10 minutes
        return [getattr(item, "label", "") for item in view.children]

    assert asyncio.run(build_view()) == ["Confirm", "Make a new note instead", "Cancel"]
