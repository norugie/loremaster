"""Answer a question from retrieved notes with the local answer model.

Steps for one question (each question stands alone, with no memory of earlier
ones):

1. Retrieve the most similar chunks (retrieve.py).
2. Build a small prompt: a system message with the rules and the vault's
   guidance note (before the notes), then a user message with the
   retrieved chunks, each labelled with its note, and the question.
3. Ask the answer model through Ollama and return its text plus the names of
   the notes whose chunks were in the prompt.

Prompt size: the answer model runs within a ~3 GB GPU budget, so its context
window (`num_ctx`) is kept small: the KV cache of Qwen3 4B grows by roughly
150 KB per token, so 2,048 tokens cost about 0.3 GB. Chunks are added best
first until `NOTES_CHAR_BUDGET`, less the guidance note's length, is reached
(the top chunk is always kept), and the reply is capped at
`MAX_ANSWER_TOKENS`. If Ollama reports a prompt that filled the context
anyway, a warning is logged, since Ollama drops the start of an over-long
prompt (the rules) rather than failing.

Everything here is synchronous (the Ollama client and the embedder block), so
the bot calls `Answerer.answer` through `asyncio.to_thread`.

Try it by hand, after an index has been built:

    docker compose run --rm app python -m loremaster.answer "your question"
"""

import argparse
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import ollama

from loremaster.chunking import clean_wikilinks
from loremaster.config import DEFAULT_MIN_SCORE, ConfigError, load_config
from loremaster.index import IndexStore, OllamaEmbedder
from loremaster.retrieve import DEFAULT_TOP_K, Hit, Retriever
from loremaster.vault import FrontmatterError, split_frontmatter

log = logging.getLogger(__name__)

# Context window for the answer model, in tokens. Keep it small (GPU memory).
CONTEXT_TOKENS = 2048
# Upper bound for the reply, in tokens (~4 characters each in English).
MAX_ANSWER_TOKENS = 350
# Characters of note text put in one prompt (~1,100 tokens of English).
NOTES_CHAR_BUDGET = 4500
# Low temperature: stick closely to the notes rather than be creative.
TEMPERATURE = 0.3

SYSTEM_RULES = """\
You answer questions about a roleplay setting's lore for members of a Discord \
server. Use only the notes you are given and the guidance below.
Rules:
- Use only facts stated in the notes. Do not add facts from your own knowledge.
- If the notes do not answer the question, say you couldn't find it in the lore.
- Answer directly in a few sentences of plain text. Do not mention note numbers."""

# Used only when no note is relevant and the config names a canon.
# "{canon}" is replaced with the config's canon name.
CANON_RULES = """\
You answer questions about a roleplay setting based on {canon}, for members \
of a Discord server. The setting's own notes say nothing about this question, \
so answer from your knowledge of the official {canon} canon.
Rules:
- State only things you are confident are official {canon} canon. Do not guess \
or invent names, events or details.
- If you don't know, or the question isn't about {canon}, say you don't know.
- Answer directly in a few sentences of plain text."""

# Reply when no note is relevant enough; the model isn't called.
NOT_FOUND_TEXT = "I couldn't find anything about that in the lore."

Message = Mapping[str, str]


class ChatModel(Protocol):
    """Anything that answers a list of chat messages. Tests use a fake one."""

    @property
    def model(self) -> str: ...

    def chat(self, messages: Sequence[Message]) -> str: ...


class OllamaChat:
    """The answer model through a local Ollama server (address from `OLLAMA_HOST`)."""

    def __init__(
        self,
        model: str,
        client: ollama.Client | None = None,
        context_tokens: int = CONTEXT_TOKENS,
        max_answer_tokens: int = MAX_ANSWER_TOKENS,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        # The timeout is passed through to httpx; a stuck model call raises
        # instead of hanging the question forever.
        self._client = client or ollama.Client(timeout=timeout)
        self._options = {
            "num_ctx": context_tokens,
            "num_predict": max_answer_tokens,
            "temperature": TEMPERATURE,
        }
        self._context_tokens = context_tokens

    @property
    def model(self) -> str:
        return self._model

    def chat(self, messages: Sequence[Message]) -> str:
        response = self._client.chat(
            model=self._model, messages=list(messages), options=self._options
        )
        used = (response.prompt_eval_count or 0) + (response.eval_count or 0)
        if used >= self._context_tokens:
            log.warning(
                "prompt + answer used %d of %d context tokens; the prompt may "
                "have been cut",
                used,
                self._context_tokens,
            )
        return (response.message.content or "").strip()


@dataclass(frozen=True)
class Answer:
    """The model's reply and the notes it was given."""

    text: str
    notes: tuple[str, ...]  # note titles, best match first, no repeats
    hits: tuple[Hit, ...]  # the chunks put in the prompt
    seconds: float = 0.0
    # False when no note was relevant enough, so the answer isn't from the notes.
    found: bool = True
    top_score: float | None = None  # best similarity retrieved, relevant or not


def load_guidance(vault: Path, note: Path | None) -> str:
    """The guidance note's text without frontmatter, wikilinks made readable.

    Returns "" when no guidance note is configured. The config loader already
    checked that the note exists; a read error here propagates.
    """
    if note is None:
        return ""
    text = (vault / note).read_text(encoding="utf-8")
    try:
        _, body = split_frontmatter(text)
    except FrontmatterError:
        body = text  # keep the guidance even if its frontmatter is broken
    return clean_wikilinks(body)[0].strip()


def relevant_hits(hits: Sequence[Hit], min_score: float) -> list[Hit]:
    """The hits scoring at least `min_score`, in their original order."""
    return [hit for hit in hits if hit.score >= min_score]


def select_hits(hits: Sequence[Hit], budget: int = NOTES_CHAR_BUDGET) -> list[Hit]:
    """The hits that fit in the prompt, best first. The first is always kept
    (chunks are at most 2,000 characters, below the budget); later ones are
    skipped if they would go over."""
    chosen: list[Hit] = []
    used = 0
    for hit in hits:
        size = len(hit.chunk.text)
        if chosen and used + size > budget:
            continue
        chosen.append(hit)
        used += size
    return chosen


def build_messages(
    question: str, hits: Sequence[Hit], guidance: str = ""
) -> list[dict[str, str]]:
    """The chat messages for one question. Pure: same input, same prompt."""
    system = SYSTEM_RULES
    if guidance:
        system += f"\n\nGuidance from the setting's staff:\n{guidance}"
    notes = "\n\n".join(
        f"[{i}] {hit.label}\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
    )
    user = f"Notes:\n\n{notes}\n\nQuestion: {question.strip()}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_canon_messages(
    question: str, canon: str, guidance: str = ""
) -> list[dict[str, str]]:
    """The chat messages for a canon-fallback answer: no notes, the guidance
    note still included."""
    system = CANON_RULES.format(canon=canon)
    if guidance:
        system += f"\n\nGuidance from the setting's staff:\n{guidance}"
    user = f"Question: {question.strip()}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def note_names(hits: Sequence[Hit]) -> tuple[str, ...]:
    """Note titles in order of first appearance, without repeats."""
    return tuple(dict.fromkeys(hit.chunk.title for hit in hits))


class Answerer:
    """Retrieves notes for a question and asks the answer model.

    Holds no conversation state: every call builds a fresh prompt from that
    question alone.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: ChatModel,
        guidance: str = "",
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        canon: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.guidance = guidance
        self.top_k = top_k
        self.min_score = min_score
        self.canon = canon  # None: vault only

    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.search(question, self.top_k)

    def select(self, hits: Sequence[Hit]) -> list[Hit]:
        """The hits that fit in the prompt. The guidance note shares the notes'
        budget, so a long one leaves fewer notes instead of overflowing the
        context (Ollama would drop the start of the prompt: the rules)."""
        return select_hits(hits, NOTES_CHAR_BUDGET - len(self.guidance))

    def answer(self, question: str, hits: Sequence[Hit] | None = None) -> Answer:
        """Answer `question` from `hits` (retrieved here if not given).

        Hits scoring below `min_score` are dropped. If none is left: with a
        canon configured, the model answers from its canon knowledge, listing no
        notes; without one, the reply is `NOT_FOUND_TEXT` and the
        model is not called. Relevant notes always win over canon
: the canon prompt is only used when there are none.
        """
        start = time.perf_counter()
        if hits is None:
            hits = self.retrieve(question)
        top_score = hits[0].score if hits else None
        relevant = relevant_hits(hits, self.min_score)
        if not relevant:
            if self.canon:
                messages = build_canon_messages(question, self.canon, self.guidance)
                text = self.model.chat(messages)
            else:
                text = NOT_FOUND_TEXT
            return Answer(
                text=text,
                notes=(),
                hits=(),
                seconds=time.perf_counter() - start,
                found=False,
                top_score=top_score,
            )
        chosen = self.select(relevant)
        text = self.model.chat(build_messages(question, chosen, self.guidance))
        return Answer(
            text=text,
            notes=note_names(chosen),
            hits=tuple(chosen),
            seconds=time.perf_counter() - start,
            top_score=top_score,
        )


def main(argv: list[str] | None = None) -> int:
    """Command-line try-it: answer one question from the index."""
    parser = argparse.ArgumentParser(
        prog="python -m loremaster.answer",
        description="Answer a question from the index with the answer model.",
    )
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=Path("lorebot.yaml"))
    parser.add_argument("--prompt", action="store_true", help="print the prompt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    if not config.index_path.is_file():
        print(f"no index at {config.index_path}; build one first", file=sys.stderr)
        return 1
    embedder = OllamaEmbedder(config.models.embedding)
    with IndexStore(config.index_path, None) as store:
        answerer = Answerer(
            Retriever(store, embedder, config.models.query_prefix),
            OllamaChat(config.models.answer),
            load_guidance(config.vault_path, config.guidance_note),
            min_score=config.retrieval.min_score,
            canon=config.canon,
        )
        hits = answerer.retrieve(args.question)
        if args.prompt:
            relevant = relevant_hits(hits, answerer.min_score)
            if relevant or not answerer.canon:
                messages = build_messages(
                    args.question, answerer.select(relevant), answerer.guidance
                )
            else:
                messages = build_canon_messages(
                    args.question, answerer.canon, answerer.guidance
                )
            for m in messages:
                print(f"--- {m['role']} ---\n{m['content']}\n")
        result = answerer.answer(args.question, hits)
    print(result.text)
    print(f"\nNotes: {', '.join(result.notes) or '(none)'}")
    scores = ", ".join(f"{h.score:.2f}" for h in hits)
    print(f"({result.seconds:.1f}s; scores {scores})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
