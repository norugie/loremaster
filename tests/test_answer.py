"""Tests for answering. Made-up notes only.

Unit tests use a fake chat model that records the prompts it gets, and a
bag-of-words fake embedder, so no Ollama is needed. The test marked
`integration` asks the real local answer model one question.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from loremaster.answer import (
    NOT_FOUND_TEXT,
    NOTES_CHAR_BUDGET,
    SYSTEM_RULES,
    WIKI_DOWN_TEXT,
    Answerer,
    Message,
    OllamaChat,
    build_messages,
    load_guidance,
    note_names,
    relevant_hits,
    select_hits,
)
from loremaster.chunking import Chunk
from loremaster.config import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_DOCUMENT_PREFIX,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_PREFIX,
    ConfigError,
    load_config,
)
from loremaster.index import IndexSettings, IndexStore, OllamaEmbedder, sync_index
from loremaster.retrieve import Hit, Retriever
from loremaster.wiki import Wiki, WikiError, WikiResult
from tests.test_retrieve import NOTES, WordsEmbedder, build

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


class FakeChat:
    """Records every prompt and answers with a fixed text."""

    def __init__(self, reply: str = "The bakery sells rye bread.") -> None:
        self.model = "fake"
        self.reply = reply
        self.prompts: list[list[Message]] = []

    def chat(self, messages: Sequence[Message]) -> str:
        self.prompts.append(list(messages))
        return self.reply


def hit(title: str, text: str, score: float = 0.8, heading: str = "") -> Hit:
    chunk = Chunk(
        note_path=Path(f"{title}.md"),
        title=title,
        headings=(heading,) if heading else (),
        index=0,
        text=text,
        links=(),
    )
    return Hit(chunk=chunk, score=score)


def answerer(tmp_path: Path, chat: FakeChat, guidance: str = "") -> Answerer:
    embedder = WordsEmbedder()
    db = build(tmp_path, NOTES, embedder)
    store = IndexStore(db, None)
    # min_score=-1: the word-count fake embedder's scores aren't calibrated.
    return Answerer(
        Retriever(store, embedder, ""), chat, guidance, top_k=2, min_score=-1
    )


# --- prompt -----------------------------------------------------------------


def test_prompt_has_rules_notes_and_question() -> None:
    messages = build_messages(
        "  What does the bakery sell? ",
        [hit("Bakery", "Rye bread and honey cakes.", heading="Goods")],
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_RULES
    user = messages[1]["content"]
    assert "[1] Bakery > Goods\nRye bread and honey cakes." in user
    assert user.endswith("Question: What does the bakery sell?")


# The guidance note's text is in every prompt, before the notes.
def test_guidance_comes_before_the_notes() -> None:
    messages = build_messages(
        "q", [hit("Bakery", "Rye bread.")], guidance="Server lore beats canon."
    )
    whole = "\n".join(m["content"] for m in messages)
    assert "Server lore beats canon." in messages[0]["content"]
    assert whole.index("Server lore beats canon.") < whole.index("Rye bread.")


def test_no_guidance_section_without_guidance() -> None:
    messages = build_messages("q", [hit("Bakery", "Rye bread.")])
    assert "Guidance" not in messages[0]["content"]


# The model is told to use only the notes.
def test_rules_forbid_outside_facts() -> None:
    assert "Use only facts stated in the notes" in SYSTEM_RULES
    assert "couldn't find it in the lore" in SYSTEM_RULES


def test_notes_are_numbered_best_first() -> None:
    user = build_messages("q", [hit("A", "first"), hit("B", "second")])[1]["content"]
    assert user.index("[1] A\nfirst") < user.index("[2] B\nsecond")


# --- choosing what goes in the prompt ----------------------------------------


def test_select_hits_keeps_order_and_budget() -> None:
    hits = [hit("A", "a" * 2000), hit("B", "b" * 2000), hit("C", "c" * 2000)]
    chosen = select_hits(hits, budget=4500)
    assert [h.chunk.title for h in chosen] == ["A", "B"]


def test_select_hits_skips_big_but_keeps_later_small() -> None:
    hits = [hit("A", "a" * 3000), hit("B", "b" * 2000), hit("C", "c" * 100)]
    assert [h.chunk.title for h in select_hits(hits, 4500)] == ["A", "C"]


def test_select_hits_always_keeps_the_top_hit() -> None:
    assert len(select_hits([hit("A", "a" * 5000)], budget=100)) == 1


def test_select_hits_empty() -> None:
    assert select_hits([]) == []


def test_default_budget_fits_two_full_chunks() -> None:
    assert NOTES_CHAR_BUDGET >= 2 * 2000


def test_note_names_dedupes_in_order() -> None:
    hits = [hit("B", "x"), hit("A", "y"), hit("B", "z")]
    assert note_names(hits) == ("B", "A")


# --- answering --------------------------------------------------------------


def test_answer_returns_text_and_notes(tmp_path: Path) -> None:
    chat = FakeChat()
    result = answerer(tmp_path, chat).answer("What bread does the bakery sell?")
    assert result.text == "The bakery sells rye bread."
    assert result.notes[0] == "bakery"
    assert len(result.hits) == 2
    assert "rye bread" in chat.prompts[0][1]["content"]


def test_answer_uses_given_hits_without_retrieving(tmp_path: Path) -> None:
    chat = FakeChat()
    result = answerer(tmp_path, chat).answer("q", [hit("Given", "given text")])
    assert result.notes == ("Given",)
    assert "given text" in chat.prompts[0][1]["content"]


def test_answer_includes_guidance(tmp_path: Path) -> None:
    chat = FakeChat()
    answerer(tmp_path, chat, guidance="Be brief.").answer("bakery bread?")
    assert "Be brief." in chat.prompts[0][0]["content"]


def test_long_guidance_leaves_fewer_notes(tmp_path: Path) -> None:
    """The guidance note shares the notes' budget, so the prompt can't overflow
    and lose its rules; the best note always stays."""
    hits = [hit("A", "a" * 2000), hit("B", "b" * 2000)]
    bot = answerer(tmp_path, FakeChat())
    assert len(bot.select(hits)) == 2
    bot.guidance = "g" * 1000
    assert [h.chunk.title for h in bot.select(hits)] == ["A"]


# Each question is answered on its own, with no memory of earlier ones.
def test_questions_do_not_share_memory(tmp_path: Path) -> None:
    chat = FakeChat()
    bot = answerer(tmp_path, chat)
    bot.answer("What bread does the bakery sell?")
    bot.answer("When is the lighthouse lamp lit?")
    second = chat.prompts[1]
    assert len(second) == 2
    assert "bakery sell" not in "\n".join(m["content"] for m in second)
    assert "The bakery sells rye bread." not in second[1]["content"]


def test_model_errors_propagate(tmp_path: Path) -> None:
    class Broken(FakeChat):
        def chat(self, messages: Sequence[Message]) -> str:
            raise TimeoutError("model timed out")

    with pytest.raises(TimeoutError):
        answerer(tmp_path, Broken()).answer("bakery?")


# --- guidance note ----------------------------------------------------------


def test_load_guidance_strips_frontmatter_and_links(tmp_path: Path) -> None:
    (tmp_path / "g.md").write_text(
        "---\nscope: guidance\n---\n# Guidance\nPrefer [[Harbor Notes|the harbor]].\n",
        encoding="utf-8",
    )
    text = load_guidance(tmp_path, Path("g.md"))
    assert text == "# Guidance\nPrefer the harbor."


def test_load_guidance_none() -> None:
    assert load_guidance(FIXTURE_VAULT, None) == ""


def test_load_guidance_fixture() -> None:
    assert "made-up guidance note" in load_guidance(FIXTURE_VAULT, Path("guidance.md"))


def _config(tmp_path: Path, guidance: str) -> Path:
    path = tmp_path / "lorebot.yaml"
    path.write_text(
        f"vault_path: {FIXTURE_VAULT.as_posix()}\n"
        "include:\n  status: [approved]\n"
        f"guidance_note: {guidance}\n",
        encoding="utf-8",
    )
    return path


# A missing guidance note stops startup with a message naming the path.
def test_missing_guidance_note_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="guidance_note: note not found.*nope.md"):
        load_config(_config(tmp_path, "nope.md"))


def test_guidance_note_outside_vault_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="guidance_note: must be inside the vault"):
        load_config(_config(tmp_path, "../test_config.py"))


def test_existing_guidance_note_loads(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, "guidance.md"))
    assert config.guidance_note == Path("guidance.md")


# --- real model -------------------------------------------------------------


@pytest.mark.integration
def test_answers_from_a_made_up_note_through_ollama(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Zorblat Bakery.md").write_text(
        "---\nstatus: approved\n---\n# Zorblat Bakery\n"
        "The Zorblat Bakery only sells violet moon-bread, baked by a baker "
        "named Quillon Ferrowind.\n",
        encoding="utf-8",
    )
    embedder = OllamaEmbedder(DEFAULT_EMBEDDING_MODEL)
    db = tmp_path / "data" / "index.sqlite"
    with IndexStore(
        db, IndexSettings(DEFAULT_EMBEDDING_MODEL, DEFAULT_DOCUMENT_PREFIX)
    ) as s:
        sync_index(vault, {"status": ["approved"]}, s, embedder)
    with IndexStore(db, None) as store:
        bot = Answerer(
            Retriever(store, embedder, DEFAULT_QUERY_PREFIX),
            OllamaChat(DEFAULT_ANSWER_MODEL),
        )
        result = bot.answer("Who bakes the bread at the Zorblat Bakery?")
    assert "Quillon" in result.text
    assert result.notes == ("Zorblat Bakery",)


# --- not found ----------------------------------------


def test_relevant_hits_keeps_scores_at_or_above_threshold() -> None:
    hits = [hit("A", "a", 0.9), hit("B", "b", 0.7), hit("C", "c", 0.69)]
    assert [h.chunk.title for h in relevant_hits(hits, 0.7)] == ["A", "B"]


# Nothing relevant enough -> "not found", and the model isn't called.
def test_nothing_relevant_says_not_found_without_the_model(tmp_path: Path) -> None:
    chat = FakeChat()
    bot = answerer(tmp_path, chat)
    bot.min_score = 0.7
    result = bot.answer("q", [hit("A", "a", 0.66), hit("B", "b", 0.5)])
    assert result.text == NOT_FOUND_TEXT
    assert result.notes == ()
    assert not result.found
    assert result.top_score == 0.66
    assert chat.prompts == []


def test_no_hits_at_all_says_not_found(tmp_path: Path) -> None:
    chat = FakeChat()
    result = answerer(tmp_path, chat).answer("q", [])
    assert result.text == NOT_FOUND_TEXT
    assert result.top_score is None
    assert chat.prompts == []


def test_low_scoring_hits_are_left_out_of_prompt_and_notes(tmp_path: Path) -> None:
    chat = FakeChat()
    bot = answerer(tmp_path, chat)
    bot.min_score = 0.7
    result = bot.answer("q", [hit("Good", "good text", 0.8), hit("Weak", "weak", 0.6)])
    assert result.found
    assert result.notes == ("Good",)
    assert "weak" not in chat.prompts[0][1]["content"]


def test_min_score_comes_from_config(tmp_path: Path) -> None:
    path = _config(tmp_path, "guidance.md")
    path.write_text(path.read_text() + "retrieval:\n  min_score: 0.65\n")
    assert load_config(path).retrieval.min_score == 0.65
    path.write_text(path.read_text().replace("0.65", "1.5"))
    with pytest.raises(ConfigError, match="retrieval.min_score"):
        load_config(path)


def test_default_min_score(tmp_path: Path) -> None:
    assert load_config(_config(tmp_path, "guidance.md")).retrieval.min_score == 0.70


# --- wiki fallback ----------------------


class FakeWiki:
    """Stands in for `Wiki`: returns fixed sections, or fails."""

    def __init__(
        self, hits: list[Hit], fail: bool = False, named: Sequence[str] = ()
    ) -> None:
        self.hits = hits
        self.fail = fail
        self.named = frozenset(named)
        self.questions: list[str] = []

    def search(self, question: str) -> WikiResult:
        self.questions.append(question)
        if self.fail:
            raise WikiError("timed out")
        urls = {
            h.chunk.title: f"https://wiki.example/{h.chunk.title}" for h in self.hits
        }
        return WikiResult(self.hits, urls, "Made-Up Wiki (CC BY-SA)", self.named)


def wiki_answerer(tmp_path: Path, chat: FakeChat, wiki: FakeWiki) -> Answerer:
    bot = answerer(tmp_path, chat)
    bot.min_score = 0.7
    # FakeWiki has Wiki's one public method; a cast keeps the test simple.
    bot.wiki = cast(Wiki, wiki)
    return bot


# No relevant note + relevant wiki sections -> answered from them, with sources.
def test_wiki_answers_when_no_note_is_relevant(tmp_path: Path) -> None:
    chat = FakeChat("The hero is Brightblade.")
    wiki = FakeWiki(
        [hit("Heroes", "Brightblade is the hero.", 0.8), hit("B", "x", 0.3)]
    )
    bot = wiki_answerer(tmp_path, chat, wiki)
    bot.canon = "Made-Up Saga"
    result = bot.answer("Who is the hero?", [hit("A", "unrelated", 0.4)])
    assert result.text == "The hero is Brightblade."
    assert result.notes == () and not result.found
    assert result.sources == (("Heroes", "https://wiki.example/Heroes"),)
    assert result.credit == "Made-Up Wiki (CC BY-SA)"
    system, user = (m["content"] for m in chat.prompts[0])
    assert "a wiki about Made-Up Saga" in system
    assert "Use only facts stated in the excerpts" in system
    assert user.startswith("Wiki excerpts:\n\n[1] Heroes\nBrightblade is the hero.")
    assert "unrelated" not in user and "[2]" not in user


# A page the question names is used even though its text scores low.
def test_a_named_wiki_page_is_used_whatever_it_scores(tmp_path: Path) -> None:
    chat = FakeChat("It is the ninth book.")
    wiki = FakeWiki(
        [hit("The Ninth Book", "The ninth book.", 0.5), hit("B", "x", 0.6)],
        named=["The Ninth Book"],
    )
    result = wiki_answerer(tmp_path, chat, wiki).answer("What is Book 9?", [])
    assert result.sources == (
        ("The Ninth Book", "https://wiki.example/The Ninth Book"),
    )
    assert "[2]" not in chat.prompts[0][1]["content"]


# Relevant notes -> the notes prompt; the wiki isn't asked.
def test_notes_win_over_the_wiki(tmp_path: Path) -> None:
    chat = FakeChat()
    wiki = FakeWiki([hit("Heroes", "wiki text", 0.9)])
    bot = wiki_answerer(tmp_path, chat, wiki)
    result = bot.answer("Who is the hero?", [hit("Heroes", "The hero is Ash.", 0.8)])
    assert result.found and result.notes == ("Heroes",) and result.sources == ()
    assert chat.prompts[0][0]["content"] == SYSTEM_RULES
    assert wiki.questions == []


# Nothing relevant on the wiki either -> not found, model not called.
def test_wiki_with_nothing_relevant_says_not_found(tmp_path: Path) -> None:
    chat = FakeChat()
    bot = wiki_answerer(tmp_path, chat, FakeWiki([hit("B", "x", 0.5)]))
    result = bot.answer("q", [])
    assert result.text == NOT_FOUND_TEXT and result.sources == ()
    assert chat.prompts == []


# Wiki out of reach -> not found (saying so), model not called, no error.
def test_wiki_failure_says_not_found(tmp_path: Path) -> None:
    chat = FakeChat()
    bot = wiki_answerer(tmp_path, chat, FakeWiki([], fail=True))
    result = bot.answer("q", [])
    assert result.text == NOT_FOUND_TEXT + WIKI_DOWN_TEXT
    assert chat.prompts == []


# A canon name alone never makes the model answer from its own knowledge.
def test_canon_without_wiki_means_vault_only(tmp_path: Path) -> None:
    chat = FakeChat()
    bot = answerer(tmp_path, chat)
    bot.min_score, bot.canon = 0.7, "Made-Up Saga"
    assert bot.answer("q", [hit("A", "a", 0.4)]).text == NOT_FOUND_TEXT
    assert chat.prompts == []


def test_canon_comes_from_config(tmp_path: Path) -> None:
    path = _config(tmp_path, "guidance.md")
    assert load_config(path).canon is None
    path.write_text(path.read_text() + "canon: Made-Up Saga\n")
    assert load_config(path).canon == "Made-Up Saga"
    path.write_text(path.read_text().replace("Made-Up Saga", '""'))
    with pytest.raises(ConfigError, match="canon"):
        load_config(path)


def test_wiki_comes_from_config(tmp_path: Path) -> None:
    path = _config(tmp_path, "guidance.md")
    assert load_config(path).wiki is None
    base = path.read_text()
    path.write_text(base + "wiki:\n  api_url: https://wiki.example/w/api.php\n")
    wiki = load_config(path).wiki
    assert wiki is not None and wiki.api_url == "https://wiki.example/w/api.php"
    for bad in ("http://wiki.example/api.php", "https://wiki.example/", "api.php"):
        path.write_text(base + f"wiki:\n  api_url: {bad}\n")
        with pytest.raises(ConfigError, match="wiki.api_url"):
            load_config(path)


def test_canon_without_wiki_logs_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _config(tmp_path, "guidance.md")
    path.write_text(path.read_text() + "canon: Made-Up Saga\n")
    load_config(path)
    assert "canon has no effect without a wiki" in caplog.text
