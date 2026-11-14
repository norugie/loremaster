"""Tests for the Discord bot's decisions.

`LoreService` holds every decision without Discord types, so these tests call
it directly with `asyncio.run` (no Discord connection, no Ollama). Connecting
to Discord is checked by hand.
"""

import asyncio
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from discord import app_commands

from loremaster.answer import Answer, Answerer, Message
from loremaster.bot import (
    CHANNEL_REFUSAL,
    COOLDOWN_SECONDS,
    DM_REFUSAL,
    ERROR_REPLY,
    Cooldown,
    LoreService,
    cooldown_message,
    find_config_path,
    format_reply,
    is_staff,
    make_lore_group,
)
from loremaster.config import Config, load_config
from loremaster.index import IndexStore
from loremaster.retrieve import Retriever
from tests.test_answer import FakeChat
from tests.test_retrieve import NOTES, WordsEmbedder, build

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


def make_config(tmp_path: Path, discord: str = "") -> Config:
    path = tmp_path / "lorebot.yaml"
    path.write_text(
        f"vault_path: {FIXTURE_VAULT.as_posix()}\n"
        "include:\n  status: [approved]\n"
        "guidance_note: guidance.md\n" + discord,
        encoding="utf-8",
    )
    return load_config(path)


def make_service(
    tmp_path: Path, chat: FakeChat | None = None, discord: str = ""
) -> LoreService:
    embedder = WordsEmbedder()
    db = build(tmp_path, NOTES, embedder)
    retriever = Retriever(IndexStore(db, None), embedder, "")
    # min_score=-1: the word-count fake embedder's scores aren't calibrated.
    answerer = Answerer(retriever, chat or FakeChat(), "", top_k=2, min_score=-1)
    return LoreService(make_config(tmp_path, discord), answerer)


def answer(text: str, notes: tuple[str, ...]) -> Answer:
    return Answer(text=text, notes=notes, hits=())


# --- pure decisions ----------------------------------------------------------


def test_format_reply_lists_notes_in_subtext() -> None:
    reply = format_reply(answer("Rye bread.", ("Bakery", "Market")))
    assert reply == "Rye bread.\n-# Notes: Bakery, Market"


def test_format_reply_without_notes() -> None:
    assert format_reply(answer("No idea.", ())) == "No idea."


def test_format_reply_empty_answer() -> None:
    assert "empty answer" in format_reply(answer("  ", ("Bakery",)))


def test_format_reply_fits_discord_limit_and_keeps_notes() -> None:
    long = "word " * 1000
    reply = format_reply(answer(long, ("Bakery",)))
    assert len(reply) <= 2000
    assert reply.endswith("…\n-# Notes: Bakery")


def test_format_reply_long_unbroken_text() -> None:
    reply = format_reply(answer("x" * 3000, ("A",)), limit=100)
    assert len(reply) <= 100
    assert reply.endswith("…\n-# Notes: A")


def test_format_reply_many_notes_still_fits() -> None:
    notes = tuple(f"Note number {i}" for i in range(300))
    reply = format_reply(answer("Short answer.", notes))
    assert len(reply) <= 2000
    assert reply.startswith("Short answer.")


def test_staff_needs_an_overlapping_role() -> None:
    assert is_staff([1, 2, 3], [3, 9])
    assert not is_staff([1, 2], [3])


# With no staff roles configured, nobody is staff.
def test_no_staff_roles_means_nobody() -> None:
    assert not is_staff([1, 2, 3], [])


def test_config_path_order() -> None:
    assert find_config_path(Path("a.yaml"), {"LOREBOT_CONFIG": "b.yaml"}) == Path(
        "a.yaml"
    )
    assert find_config_path(None, {"LOREBOT_CONFIG": "b.yaml"}) == Path("b.yaml")
    assert find_config_path(None, {"LOREBOT_CONFIG": "  "}) == Path("lorebot.yaml")
    assert find_config_path(None, {}) == Path("lorebot.yaml")


# --- where the bot answers --------------------------------------------------


def test_refuses_in_direct_messages(tmp_path: Path) -> None:
    refusal = make_service(tmp_path).admit(guild_id=None, channel_id=5)
    assert refusal is not None and refusal.message == DM_REFUSAL


# No channel list means any channel in any server.
def test_empty_channel_list_allows_any_server_and_channel(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    assert service.admit(guild_id=1, channel_id=10) is None
    assert service.admit(guild_id=2, channel_id=99) is None


# With a list, only those channels, whichever server they're in.
def test_channel_list_limits_channels(tmp_path: Path) -> None:
    service = make_service(
        tmp_path, discord="discord:\n  allowed_channel_ids: [10, 20]\n"
    )
    assert service.admit(guild_id=1, channel_id=10) is None
    assert service.admit(guild_id=2, channel_id=20) is None
    refusal = service.admit(guild_id=1, channel_id=30)
    assert refusal is not None and refusal.message == CHANNEL_REFUSAL


# --- answering --------------------------------------------------------------


# The reply has the answer and the names of the notes used.
def test_ask_replies_with_answer_and_notes(tmp_path: Path) -> None:
    reply = asyncio.run(make_service(tmp_path).ask("What bread does the bakery sell?"))
    assert reply.startswith(
        "> What bread does the bakery sell?\nThe bakery sells rye bread."
    )
    assert "-# Notes: bakery" in reply


# A model error becomes a short reply, and the next question still works.
def test_model_error_becomes_error_reply(tmp_path: Path) -> None:
    class Flaky(FakeChat):
        def chat(self, messages: Sequence[Message]) -> str:
            if not self.prompts:
                self.prompts.append(list(messages))
                raise TimeoutError("model timed out")
            return super().chat(messages)

    service = make_service(tmp_path, Flaky())

    async def two_questions() -> tuple[str, str]:
        return await service.ask("bakery?"), await service.ask("bakery bread?")

    first, second = asyncio.run(two_questions())
    assert first == ERROR_REPLY
    assert second.startswith("> bakery bread?\nThe bakery sells rye bread.")


def test_retrieval_error_becomes_error_reply(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    # An empty question makes the retriever raise ValueError.
    assert asyncio.run(service.ask("   ")) == ERROR_REPLY


def test_answering_runs_off_the_event_loop(tmp_path: Path) -> None:
    threads: list[str] = []

    class Recording(FakeChat):
        def chat(self, messages: Sequence[Message]) -> str:
            threads.append(threading.current_thread().name)
            return super().chat(messages)

    asyncio.run(make_service(tmp_path, Recording()).ask("bakery?"))
    assert threads and threads[0] != threading.main_thread().name


def test_one_model_call_at_a_time(tmp_path: Path) -> None:
    active = 0
    most = 0
    lock = threading.Lock()

    class Slow(FakeChat):
        def chat(self, messages: Sequence[Message]) -> str:
            nonlocal active, most
            with lock:
                active += 1
                most = max(most, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return super().chat(messages)

    service = make_service(tmp_path, Slow())

    async def three() -> list[str]:
        return list(await asyncio.gather(*(service.ask("bakery?") for _ in range(3))))

    replies = asyncio.run(three())
    assert len(replies) == 3
    assert most == 1


def test_reload_guidance_reads_the_note(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.reload_guidance()
    assert "made-up guidance note" in service.answerer.guidance


def test_reload_guidance_keeps_old_text_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    service.answerer.guidance = "old"

    def broken(_vault: Path, _note: Path | None) -> str:
        raise OSError("gone")

    monkeypatch.setattr("loremaster.bot.load_guidance", broken)
    service.reload_guidance()
    assert service.answerer.guidance == "old"


# --- the slash command ------------------------------------------------------


# `/lore ask` takes the question as text.
def test_lore_group_has_ask_with_a_text_question(tmp_path: Path) -> None:
    group = make_lore_group(make_service(tmp_path))
    assert group.name == "lore"
    ask = group.get_command("ask")
    assert isinstance(ask, app_commands.Command)
    [param] = ask.parameters
    assert param.name == "question"
    assert param.required
    assert param.max_value == 300  # Range[str, 1, 300]: Discord limits its length


# --- cooldown ----------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def cooled_service(
    tmp_path: Path, chat: FakeChat | None = None
) -> tuple[LoreService, FakeClock]:
    clock = FakeClock()
    service = make_service(tmp_path, chat)
    service.cooldown = Cooldown(COOLDOWN_SECONDS, clock)
    return service, clock


def test_cooldown_is_15_seconds() -> None:
    assert COOLDOWN_SECONDS == 15


def test_cooldown_message_rounds_up() -> None:
    assert cooldown_message(12.2) == "On cooldown: try again in 13 seconds."
    assert cooldown_message(0.3) == "On cooldown: try again in 1 second."


def test_cooldown_counts_down() -> None:
    clock = FakeClock()
    cooldown = Cooldown(15, clock)
    assert cooldown.remaining(1) == 0
    cooldown.start(1)
    clock.now += 10
    assert cooldown.remaining(1) == 5
    clock.now += 5
    assert cooldown.remaining(1) == 0


# A second question in the same server within 15 s is refused with the
# seconds left, and the model isn't called.
def test_second_question_in_same_server_is_refused(tmp_path: Path) -> None:
    chat = FakeChat()
    service, clock = cooled_service(tmp_path, chat)
    assert service.admit(guild_id=1, channel_id=10) is None
    asyncio.run(service.ask("bakery?", guild_id=1))
    clock.now += 3
    refusal = service.admit(guild_id=1, channel_id=11)  # another channel, same server
    assert refusal is not None
    assert refusal.message == "On cooldown: try again in 12 seconds."
    assert len(chat.prompts) == 1


def test_cooldown_ends_after_15_seconds(tmp_path: Path) -> None:
    service, clock = cooled_service(tmp_path)
    assert service.admit(guild_id=1, channel_id=10) is None
    asyncio.run(service.ask("bakery?", guild_id=1))
    clock.now += 15
    assert service.admit(guild_id=1, channel_id=10) is None


# Each server has its own cooldown.
def test_other_servers_are_not_affected(tmp_path: Path) -> None:
    service, _ = cooled_service(tmp_path)
    assert service.admit(guild_id=1, channel_id=10) is None
    assert service.admit(guild_id=2, channel_id=20) is None
    assert service.admit(guild_id=1, channel_id=10) is not None


def test_question_still_answering_blocks_the_next(tmp_path: Path) -> None:
    service, _ = cooled_service(tmp_path)
    assert service.admit(guild_id=1, channel_id=10) is None
    # the first answer hasn't finished; a second question arrives at once
    assert service.admit(guild_id=1, channel_id=10) is not None


def test_cooldown_restarts_when_the_answer_finishes(tmp_path: Path) -> None:
    class SlowClockChat(FakeChat):
        def chat(self, messages: Sequence[Message]) -> str:
            clock.now += 10  # the model takes 10 s
            return super().chat(messages)

    service, clock = cooled_service(tmp_path, SlowClockChat())
    assert service.admit(guild_id=1, channel_id=10) is None
    asyncio.run(service.ask("bakery?", guild_id=1))
    clock.now += 14
    assert service.admit(guild_id=1, channel_id=10) is not None  # 1 s left
    clock.now += 1
    assert service.admit(guild_id=1, channel_id=10) is None


def test_refused_questions_do_not_start_a_cooldown(tmp_path: Path) -> None:
    service, _ = cooled_service(tmp_path)
    service.config = make_config(
        tmp_path, discord="discord:\n  allowed_channel_ids: [10]\n"
    )
    assert service.admit(guild_id=1, channel_id=99) is not None  # wrong channel
    assert service.admit(guild_id=1, channel_id=10) is None


def test_failed_answer_still_starts_cooldown(tmp_path: Path) -> None:
    class Broken(FakeChat):
        def chat(self, messages: Sequence[Message]) -> str:
            raise TimeoutError

    service, clock = cooled_service(tmp_path, Broken())
    assert service.admit(guild_id=1, channel_id=10) is None
    assert asyncio.run(service.ask("bakery?", guild_id=1)) == ERROR_REPLY
    clock.now += 5
    assert service.admit(guild_id=1, channel_id=10) is not None


# The question asked is shown as a quote line above the answer.
def test_format_reply_shows_the_question() -> None:
    reply = format_reply(answer("Rye bread.", ("Bakery",)), "What  does it\nsell?")
    assert reply == "> What does it sell?\nRye bread.\n-# Notes: Bakery"


def test_format_reply_keeps_question_and_notes_when_cutting() -> None:
    reply = format_reply(answer("word " * 1000, ("Bakery",)), "q" * 300)
    assert len(reply) <= 2000
    assert reply.startswith("> " + "q" * 300 + "\n")
    assert reply.endswith("…\n-# Notes: Bakery")


# A wiki answer says it isn't server lore and credits and links the pages.
def test_format_reply_credits_the_wiki() -> None:
    wiki_answer = Answer(
        text="word " * 1000,
        notes=(),
        hits=(),
        found=False,
        sources=(("Harbor [Guild]", "https://wiki.example/wiki/Harbor_(Guild)"),),
        credit="Made-Up Wiki (CC BY-SA)",
    )
    reply = format_reply(wiki_answer, "q")
    assert len(reply) <= 2000
    assert reply.endswith(
        "…\n-# Not in the server lore. From Made-Up Wiki (CC BY-SA): "
        "[Harbor \\[Guild\\]](<https://wiki.example/wiki/Harbor_(Guild)>)"
    )
