"""Tests for the wiki fallback. No network: a fake fetch function returns
canned API responses for a made-up wiki."""

import json
import urllib.parse
from pathlib import Path

import pytest

from loremaster.wiki import (
    CACHE_SECONDS,
    MAX_PAGE_CHUNKS,
    Wiki,
    WikiError,
    html_sections,
    main,
    page_chunks,
)
from tests.test_retrieve import WordsEmbedder

API = "https://wiki.example/w/api.php"

BAKERY_HTML = """
<div class="mw-parser-output">
<aside class="portable-infobox"><h2>Bakery</h2><h3>Owner</h3><div>Quillon</div></aside>
<p>The <a href="/wiki/Bakery">bakery</a> sells rye<sup class="reference">[1]</sup>
bread.</p>
<script>alert("no")</script>
<h2><span class="mw-headline">Goods</span>
<span class="mw-editsection">[edit]</span></h2>
<ul><li>Honey cakes</li><li>Rye<br>loaves</li></ul>
<h3>Prices</h3>
<p>One  copper each.</p>
<h2>History</h2>
<h2>See also</h2>
<div class="navbox"><p>Other shops</p></div>
</div>
"""


def search_json(*pages: tuple[int, str, str], license_name: str = "CC BY-SA") -> str:
    return json.dumps(
        {
            "query": {
                "pages": [
                    {"pageid": i, "title": t, "index": rank, "canonicalurl": url}
                    for rank, (i, t, url) in enumerate(pages, start=1)
                ],
                "general": {"sitename": "Made-Up Wiki"},
                "rightsinfo": {"text": license_name},
            }
        }
    )


class FakeFetch:
    """Answers API URLs from canned data and records them."""

    def __init__(self, search: str, pages: dict[int, str]) -> None:
        self.search = search
        self.pages = pages  # page id -> HTML; a missing id gives an API error
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        base, _, query = url.partition("?")
        assert base == API
        params = dict(urllib.parse.parse_qsl(query))
        assert params["format"] == "json" and params["formatversion"] == "2"
        if params["action"] == "query":
            return self.search.encode()
        html = self.pages.get(int(params["pageid"]))
        if html is None:
            return b'{"error": {"code": "missingtitle"}}'
        return json.dumps({"parse": {"text": html}}).encode()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make(fetch: FakeFetch, clock: FakeClock | None = None) -> Wiki:
    return Wiki(API, WordsEmbedder(), fetch=fetch, clock=clock or FakeClock())


BAKERY = (1, "Bakery", "https://wiki.example/wiki/Bakery")
MILL = (2, "Mill", "https://wiki.example/wiki/Mill")


# --- HTML to sections --------------------------------------------------------


def test_html_is_split_at_headings_and_cleaned() -> None:
    assert html_sections(BAKERY_HTML) == [
        ((), "Bakery\nOwner\nQuillon\nThe bakery sells rye bread."),
        (("Goods",), "Honey cakes\nRye\nloaves"),
        (("Goods", "Prices"), "One copper each."),
    ]


def test_long_pages_are_capped_and_long_sections_split() -> None:
    chunks = page_chunks("Big", "<p>" + "word " * 1000 + "</p>")
    assert len(chunks) == 3 and all(len(c.text) <= 2000 for c in chunks)
    assert [c.index for c in chunks] == [0, 1, 2]
    many = "".join(f"<h2>S{i}</h2><p>text</p>" for i in range(60))
    assert len(page_chunks("Many", many)) == MAX_PAGE_CHUNKS


# --- search ------------------------------------------------------------------


def test_search_scores_sections_and_credits_the_wiki() -> None:
    fetch = FakeFetch(
        search_json(BAKERY, MILL),
        {1: BAKERY_HTML, 2: "<p>The mill grinds wheat into flour.</p>"},
    )
    result = make(fetch).search("What do honey cakes cost at the bakery?")
    assert result.credit == "Made-Up Wiki (CC BY-SA)"
    assert result.urls == {"Bakery": BAKERY[2], "Mill": MILL[2]}
    assert result.hits[0].chunk.title == "Bakery"
    assert result.hits[-1].chunk.title == "Mill"
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert len(fetch.urls) == 3  # one search, two pages
    assert "gsrsearch=What+do+honey+cakes" in fetch.urls[0]


def test_no_search_results_means_no_hits_and_no_page_requests() -> None:
    fetch = FakeFetch(search_json(license_name=""), {})
    result = make(fetch).search("anything")
    assert result.hits == [] and result.credit == "Made-Up Wiki"
    assert len(fetch.urls) == 1


def test_pages_on_another_site_are_ignored() -> None:
    evil = (3, "Evil", "https://elsewhere.example/wiki/Evil")
    fetch = FakeFetch(search_json(evil, BAKERY), {1: BAKERY_HTML, 3: "<p>x</p>"})
    assert list(make(fetch).search("bakery").urls) == ["Bakery"]


def test_a_failing_page_is_skipped() -> None:
    fetch = FakeFetch(search_json(MILL, BAKERY), {1: BAKERY_HTML})  # no Mill page
    assert list(make(fetch).search("bakery").urls) == ["Bakery"]


@pytest.mark.parametrize(
    "body", ["<html>Just a moment...</html>", '{"error": {"code": "ratelimited"}}', ""]
)
def test_bad_search_responses_raise_wiki_error(body: str) -> None:
    with pytest.raises(WikiError):
        make(FakeFetch(body, {})).search("q")


def test_fetch_errors_propagate_as_wiki_error() -> None:
    def broken(url: str) -> bytes:
        raise WikiError("request failed: timed out")

    with pytest.raises(WikiError, match="timed out"):
        Wiki(API, WordsEmbedder(), fetch=broken).search("q")


# A page named in the question (by title or redirect) comes first, even when
# another page's text looks more similar to the question.
def test_a_page_named_in_the_question_comes_first() -> None:
    search = json.loads(search_json(MILL, BAKERY))
    search["query"]["pages"][1]["redirects"] = [{"title": "Shop 9"}, {"title": "S9"}]
    fetch = FakeFetch(
        json.dumps(search),
        {1: "<p>Rye bread.</p>", 2: "<p>What happened at shop nine, the mill?</p>"},
    )
    result = make(fetch).search("What happened at Shop 9?")
    assert result.named == {"Bakery"}
    assert [h.chunk.title for h in result.hits] == ["Bakery", "Mill"]
    assert result.hits[0].score < result.hits[1].score
    assert "prop=info%7Credirects" in fetch.urls[0]
    assert make(fetch).search("the workshop 99?").named == set()  # whole words only


# --- cache -------------------------------------------------------------------


def test_repeat_questions_are_served_from_the_cache_for_a_day() -> None:
    fetch = FakeFetch(search_json(BAKERY), {1: BAKERY_HTML})
    clock = FakeClock()
    wiki = make(fetch, clock)
    first = wiki.search("What does the bakery sell?")
    again = wiki.search("  what does the BAKERY sell? ")
    assert len(fetch.urls) == 2  # nothing new was fetched
    assert [h.chunk for h in again.hits] == [h.chunk for h in first.hits]
    clock.now = CACHE_SECONDS + 1
    wiki.search("What does the bakery sell?")
    assert len(fetch.urls) == 4


def test_a_cached_page_is_reused_by_another_question() -> None:
    fetch = FakeFetch(search_json(BAKERY), {1: BAKERY_HTML})
    wiki = make(fetch)
    wiki.search("bakery goods")
    wiki.search("bakery prices")
    assert len(fetch.urls) == 3  # two searches, one page fetch


# --- command line ------------------------------------------------------------


def test_cli_without_a_wiki_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vault = Path(__file__).parent / "fixtures" / "vault"
    config = tmp_path / "lorebot.yaml"
    config.write_text(
        f"vault_path: {vault.as_posix()}\ninclude:\n  status: [approved]\n"
    )
    assert main(["q", "--config", str(config)]) == 1
    assert "no wiki block" in capsys.readouterr().err
