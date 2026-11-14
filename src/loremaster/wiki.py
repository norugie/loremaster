"""Find text about a question on a MediaWiki wiki (the fallback when no vault
note is relevant).

Steps for one question:

1. Search: one API request returns the best-matching article titles with their
   page URLs, plus the wiki's name and license (for the credit line).
2. Fetch each page as rendered HTML, one request at a time, and turn it into
   plain-text sections by heading, like a note's chunks.
3. Embed the sections with the same embedding model as the vault and score
   them against the question. The caller keeps the ones that are relevant.

The wiki's own search matches keywords; step 3 is what judges meaning.

Wiki text is untrusted: it is only ever put in the prompt as excerpts, and
page links are shown only when they point at the configured wiki.

Network: the standard library's `urllib.request` (like `fetch`, but blocking;
fine here because answering already runs in a worker thread). The function
that fetches a URL is a parameter, so tests pass a fake and never touch the
network. Any failure raises `WikiError`.

Try it by hand, with a `wiki` block in the config:

    docker compose run --rm app python -m loremaster.wiki "your question"
"""

import argparse
import http.client
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import ollama
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from loremaster import __version__
from loremaster.chunking import MAX_CHUNK_CHARS, Chunk, _split_to_size
from loremaster.config import Config, ConfigError, load_config
from loremaster.index import Embedder, OllamaEmbedder, embedding_input, normalize
from loremaster.retrieve import Hit

log = logging.getLogger(__name__)

# No personal details. MediaWiki's API etiquette suggests contact information.
USER_AGENT = f"Loremaster/{__version__} (Discord lore bot)"
TIMEOUT_S = 5.0  # per request
MAX_BYTES = 2_000_000  # per response, after decompression
MAX_PAGES = 3  # pages read per question
# shortcut: only the first 40 sections of a page are embedded (CPU time);
# raise it if answers sit further down long pages.
MAX_PAGE_CHUNKS = 40
CACHE_SECONDS = 24 * 60 * 60.0
CACHE_ENTRIES = 100

Fetch = Callable[[str], bytes]


class WikiError(Exception):
    """The wiki could not be reached or sent something unexpected."""


def http_get(url: str) -> bytes:
    """GET `url` and return the body, at most `MAX_BYTES`."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            data: bytes = response.read(MAX_BYTES + 1)
            if response.headers.get("Content-Encoding") == "gzip":
                # wbits=31: gzip format. The size cap also stops a "zip bomb".
                data = zlib.decompressobj(31).decompress(data, MAX_BYTES + 1)
    # OSError covers DNS failures, refused connections, timeouts and HTTP
    # error statuses (urllib's errors are subclasses of it).
    except (OSError, http.client.HTTPException, zlib.error) as e:
        raise WikiError(f"request failed: {e}") from e
    if len(data) > MAX_BYTES:
        raise WikiError("response too large")
    return data


# --- API responses (only the fields used; pydantic ignores the rest) ---------


class _Redirect(BaseModel):
    title: str


class _Page(BaseModel):
    pageid: int
    title: str
    index: int = 0  # rank in the search results, from 1
    canonicalurl: str = ""
    redirects: list[_Redirect] = []  # other names that lead to this page


class _General(BaseModel):
    sitename: str = ""


class _Rights(BaseModel):
    text: str = ""


class _Query(BaseModel):
    pages: list[_Page] = []
    general: _General = _General()
    rightsinfo: _Rights = _Rights()


class _SearchResponse(BaseModel):
    error: dict[str, object] | None = None
    query: _Query = _Query()


class _Parse(BaseModel):
    text: str


class _ParseResponse(BaseModel):
    parse: _Parse


# --- HTML to sections --------------------------------------------------------

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"}
_VOID |= {"source", "track", "wbr"}
_SKIP_TAGS = {"script", "style", "sup", "figure", "nav", "noscript"}
# Class-name fragments of page furniture: navigation boxes, tables of
# contents, edit links, reference lists.
_SKIP_CLASSES = ("navbox", "toc", "mw-editsection", "reference", "noprint")
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# Headings inside these are layout (e.g. an infobox's labels), not sections.
_BOXES = {"aside", "table"}
_INLINE = {"a", "b", "i", "em", "strong", "span", "small", "u", "code", "abbr"}
_INLINE |= {"ruby", "rb", "rt", "rp", "big", "font", "s", "sub", "q", "cite", "bdi"}


class _SectionParser(HTMLParser):
    """Collects a page's visible text, split at its headings.

    `HTMLParser` calls the `handle_*` methods as it reads tags and text, like
    a SAX parser; this class only keeps track of where it is.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[tuple[tuple[str, ...], str]] = []
        self._stack: list[tuple[int, str]] = []  # open headings: (level, text)
        self._text: list[str] = []
        self._skip = 0  # depth inside a skipped element; 0 = not skipping
        self._box = 0  # depth inside an aside or table
        self._heading: int | None = None  # level of the heading being read
        self._heading_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID:
            if tag == "br":
                self._break()
            return
        classes = dict(attrs).get("class") or ""
        if self._skip or tag in _SKIP_TAGS or any(c in classes for c in _SKIP_CLASSES):
            self._skip += 1
            return
        if tag in _BOXES:
            self._box += 1
        if tag in _HEADINGS and not self._box:
            self._heading = _HEADINGS[tag]
            self._heading_text = []
        elif tag not in _INLINE:
            self._break()

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if self._skip:
            self._skip -= 1
            return
        if tag in _BOXES and self._box:
            self._box -= 1
        if tag in _HEADINGS and self._heading is not None:
            self._flush()
            level, self._heading = self._heading, None
            while self._stack and self._stack[-1][0] >= level:
                self._stack.pop()
            self._stack.append((level, " ".join("".join(self._heading_text).split())))
        elif tag not in _INLINE:
            self._break()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        # Line breaks in the HTML source are just spaces; only tags break lines.
        data = data.replace("\n", " ")
        (self._heading_text if self._heading is not None else self._text).append(data)

    def _break(self) -> None:
        if not self._skip and self._heading is None:
            self._text.append("\n")

    def _flush(self) -> None:
        lines = (" ".join(line.split()) for line in "".join(self._text).splitlines())
        text = "\n".join(line for line in lines if line)
        if text:
            self.sections.append((tuple(h for _, h in self._stack), text))
        self._text = []

    def close(self) -> None:
        super().close()
        self._flush()


def html_sections(html: str) -> list[tuple[tuple[str, ...], str]]:
    """`(heading path, text)` for each section of a rendered page that has
    text. Text before the first heading has an empty heading path."""
    parser = _SectionParser()
    parser.feed(html)
    parser.close()
    return parser.sections


def page_chunks(title: str, html: str) -> list[Chunk]:
    """A page's sections as chunks, like a note's (long sections are split)."""
    chunks: list[Chunk] = []
    for headings, text in html_sections(html):
        for piece in _split_to_size(text, MAX_CHUNK_CHARS):
            chunks.append(Chunk(Path(title), title, headings, len(chunks), piece, ()))
    return chunks[:MAX_PAGE_CHUNKS]


# --- the wiki ----------------------------------------------------------------


def _words(text: str) -> str:
    """Lower-case words with single spaces around each: " like this "."""
    return f" {' '.join(re.findall(r'\w+', text.lower()))} "


def names_page(question: str, page: _Page) -> bool:
    """Whether the question contains the page's title or a redirect to it,
    as whole words. One- and two-character names are ignored."""
    asked = _words(question)
    names = [page.title, *(r.title for r in page.redirects)]
    return any(len(n) > 2 and _words(n) in asked for n in names)


@dataclass(frozen=True)
class WikiResult:
    """What a search found: sections scored against the question. Sections of
    pages the question names come first, then the rest, each best first."""

    hits: list[Hit]
    urls: dict[str, str]  # page title -> page URL, for the pages in `hits`
    credit: str  # e.g. "Some Wiki (CC BY-SA)"
    # Titles of pages the question names, by title or redirect ("Book 9" for
    # a page titled "The Ninth Book"). The embedding model can't know such
    # names mean the same thing, so these pages count as relevant whatever
    # their sections score.
    named: frozenset[str] = frozenset()


class Wiki:
    """Searches one wiki. Pages and searches are remembered for a day, so a
    repeated question costs the wiki nothing. Not thread-safe: the bot answers
    one question at a time."""

    def __init__(
        self,
        api_url: str,
        embedder: Embedder,
        document_prefix: str = "",
        query_prefix: str = "",
        fetch: Fetch = http_get,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api_url = api_url
        self._embedder = embedder
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._fetch = fetch
        self._clock = clock
        parts = urllib.parse.urlsplit(api_url)
        self._origin = f"{parts.scheme}://{parts.netloc}/"
        self._host = parts.netloc
        # Both caches: key -> (expiry time, value).
        self._searches: dict[str, tuple[float, tuple[list[_Page], str]]] = {}
        self._pages: dict[int, tuple[float, tuple[list[Chunk], NDArray[np.float32]]]]
        self._pages = {}

    def _api[T: BaseModel](self, model: type[T], **params: str) -> T:
        """One API request, validated into `model`."""
        query = urllib.parse.urlencode(
            {**params, "format": "json", "formatversion": "2"}
        )
        try:
            return model.model_validate_json(self._fetch(f"{self.api_url}?{query}"))
        except ValidationError as e:
            # Not the JSON we asked for: an API error, or a block page.
            raise WikiError(f"unexpected response to action={params['action']}") from e

    def _find_pages(self, question: str) -> tuple[list[_Page], str]:
        """The best-matching pages on this wiki, and the credit line."""
        response = self._api(
            _SearchResponse,
            action="query",
            generator="search",
            gsrsearch=question,
            gsrlimit=str(MAX_PAGES),
            gsrnamespace="0",  # articles only
            prop="info|redirects",
            inprop="url",
            rdprop="title",
            rdlimit="max",
            meta="siteinfo",
            siprop="general|rightsinfo",
        )
        if response.error is not None:
            raise WikiError(f"API error: {response.error.get('code')}")
        query = response.query
        pages = sorted(
            (p for p in query.pages if p.canonicalurl.startswith(self._origin)),
            key=lambda p: p.index,
        )
        name = query.general.sitename or self._host
        license_name = query.rightsinfo.text
        return pages, f"{name} ({license_name})" if license_name else name

    def _load_page(self, page: _Page) -> tuple[list[Chunk], NDArray[np.float32]]:
        """A page's chunks and their vectors (each scaled to length 1)."""
        response = self._api(
            _ParseResponse,
            action="parse",
            pageid=str(page.pageid),
            prop="text",
            disabletoc="1",
            disableeditsection="1",
            disablelimitreport="1",
        )
        chunks = page_chunks(page.title, response.parse.text)
        inputs = [embedding_input(c, self._document_prefix) for c in chunks]
        return chunks, normalize(self._embedder.embed(inputs) if inputs else [])

    def _cached[K, V](
        self, cache: dict[K, tuple[float, V]], key: K, load: Callable[[], V]
    ) -> V:
        now = self._clock()
        entry = cache.get(key)
        if entry is not None and entry[0] > now:
            return entry[1]
        value = load()
        cache.pop(key, None)  # so a refreshed entry counts as the newest
        cache[key] = (now + CACHE_SECONDS, value)
        if len(cache) > CACHE_ENTRIES:
            del cache[next(iter(cache))]  # dicts keep insertion order: the oldest
        return value

    def search(self, question: str) -> WikiResult:
        """Sections of the best-matching pages, scored against `question`.

        Raises `WikiError` if the search fails. A page that fails to load or
        embed is logged and skipped.
        """
        key = " ".join(question.lower().split())
        pages, credit = self._cached(
            self._searches, key, lambda: self._find_pages(question)
        )
        hits: list[Hit] = []
        urls: dict[str, str] = {}
        if not pages:
            return WikiResult(hits, urls, credit)
        asked = self._embedder.embed([self._query_prefix + question.strip()])
        vector = normalize(asked)[0]
        for page in pages:
            try:
                chunks, vectors = self._cached(
                    self._pages, page.pageid, partial(self._load_page, page)
                )
            # ResponseError: text too dense for the embedder's context.
            except (WikiError, ollama.ResponseError) as e:
                log.warning("wiki page %r skipped: %s", page.title, e)
                continue
            if chunks:
                urls[page.title] = page.canonicalurl
                scores = vectors @ vector
                hits += [Hit(c, float(s)) for c, s in zip(chunks, scores, strict=True)]
        named = frozenset(p.title for p in pages if names_page(question, p))
        hits.sort(key=lambda h: (h.chunk.title in named, h.score), reverse=True)
        return WikiResult(hits, urls, credit, named)


def make_wiki(config: Config, embedder: Embedder) -> Wiki | None:
    """The configured wiki, or None when the config has no `wiki` block."""
    if config.wiki is None:
        return None
    return Wiki(
        config.wiki.api_url,
        embedder,
        config.models.document_prefix,
        config.models.query_prefix,
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line try-it: print the best wiki sections and their scores."""
    parser = argparse.ArgumentParser(
        prog="python -m loremaster.wiki",
        description="Show the wiki sections most similar to a question.",
    )
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=8, help="number of sections")
    parser.add_argument("--config", type=Path, default=Path("lorebot.yaml"))
    parser.add_argument("--text", action="store_true", help="also print the text")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    wiki = make_wiki(config, OllamaEmbedder(config.models.embedding))
    if wiki is None:
        print("the config has no wiki block", file=sys.stderr)
        return 1
    start = time.perf_counter()
    try:
        result = wiki.search(args.question)
    except WikiError as e:
        print(f"wiki unavailable: {e}", file=sys.stderr)
        return 1
    took = time.perf_counter() - start
    for rank, hit in enumerate(result.hits[: args.k], start=1):
        print(f"{rank}. {hit.score:.3f}  {hit.label}")
        if args.text:
            print(f"   {hit.chunk.text}\n")
    print(f"pages: {', '.join(result.urls) or '(none)'}")
    print(f"{len(result.hits)} sections from {result.credit} in {took:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
