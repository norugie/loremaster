"""Tests for chunking and wikilinks. Made-up notes."""

from pathlib import Path

import pytest

from loremaster.chunking import Chunk, chunk_note, clean_wikilinks
from loremaster.vault import Note, load_notes

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


def make(body: str, path: str = "Places/sample-town.md") -> Note:
    return Note(path=Path(path), frontmatter={}, body=body)


def summary(chunks: list[Chunk]) -> list[tuple[tuple[str, ...], str]]:
    return [(c.headings, c.text) for c in chunks]


# --- headings -------------------------------------------------------------------


def test_splits_by_heading_with_heading_path() -> None:
    body = (
        "Intro before any heading.\n"
        "# Sample Town\n"
        "Town text.\n"
        "## Market\n"
        "Market text.\n"
        "### Stalls\n"
        "Stall text.\n"
        "## Docks\n"
        "Dock text.\n"
        "# Second Top\n"
        "More.\n"
    )
    chunks = chunk_note(make(body))
    assert summary(chunks) == [
        ((), "Intro before any heading."),
        (("Sample Town",), "Town text."),
        (("Sample Town", "Market"), "Market text."),
        (("Sample Town", "Market", "Stalls"), "Stall text."),
        (("Sample Town", "Docks"), "Dock text."),
        (("Second Top",), "More."),
    ]
    assert [c.index for c in chunks] == [0, 1, 2, 3, 4, 5]
    assert chunks[3].label == "sample-town > Sample Town > Market > Stalls"
    assert all(c.note_path == Path("Places/sample-town.md") for c in chunks)
    assert all(c.title == "sample-town" for c in chunks)


def test_note_without_headings_is_one_chunk() -> None:
    chunks = chunk_note(make("Just a paragraph.\n\nAnd another.\n"))
    assert summary(chunks) == [((), "Just a paragraph.\n\nAnd another.")]


def test_empty_sections_make_no_chunks() -> None:
    body = "\n# Title\n\n## Empty\n   \n## Full\nText.\n"
    assert summary(chunk_note(make(body))) == [(("Title", "Full"), "Text.")]


def test_empty_note_makes_no_chunks() -> None:
    assert chunk_note(make("")) == []


def test_skipped_heading_levels() -> None:
    body = "# A\n### Deep\nText.\n## B\nMore.\n"
    assert summary(chunk_note(make(body))) == [
        (("A", "Deep"), "Text."),
        (("A", "B"), "More."),
    ]


@pytest.mark.parametrize("line", ["#hashtag text", "####### seven", "    # indented"])
def test_not_headings(line: str) -> None:
    assert summary(chunk_note(make(f"{line}\n"))) == [((), line.strip())]


def test_closing_hashes_are_removed() -> None:
    assert chunk_note(make("## Title ##\nText.\n"))[0].headings == ("Title",)


def test_headings_inside_code_fences_do_not_split() -> None:
    body = (
        "# Real\n"
        "```md\n"
        "# not a heading\n"
        "~~~\n"
        "## still code\n"
        "```\n"
        "~~~~\n"
        "# also code\n"
        "~~~~\n"
        "After.\n"
    )
    chunks = chunk_note(make(body))
    assert len(chunks) == 1
    assert chunks[0].headings == ("Real",)
    assert "# not a heading" in chunks[0].text
    assert "# also code" in chunks[0].text


def test_unclosed_fence_runs_to_end() -> None:
    chunks = chunk_note(make("# A\n```\n# code\n"))
    assert summary(chunks) == [(("A",), "```\n# code")]


def test_crlf_body() -> None:
    chunks = chunk_note(make("# A\r\nOne.\r\n## B\r\nTwo.\r\n"))
    assert summary(chunks) == [(("A",), "One."), (("A", "B"), "Two.")]


# --- size limits ------------------------------------------------------------------


def test_long_section_splits_at_paragraphs_without_losing_text() -> None:
    paragraphs = [f"Paragraph {i}" + " word" * 15 for i in range(10)]
    body = "# Long\n" + "\n\n".join(paragraphs) + "\n"
    chunks = chunk_note(make(body), max_chars=200)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.headings == ("Long",) for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Every paragraph survives whole, in order.
    assert "\n\n".join(c.text for c in chunks) == "\n\n".join(
        p.strip() for p in paragraphs
    )


def test_one_huge_paragraph_splits_at_words() -> None:
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_note(make(" ".join(words)), max_chars=100)
    assert all(len(c.text) <= 100 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == words


def test_unbroken_text_splits_by_characters() -> None:
    chunks = chunk_note(make("x" * 250), max_chars=100)
    assert [len(c.text) for c in chunks] == [100, 100, 50]


# --- wikilinks --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "clean", "links"),
    [
        ("See [[Old Mill]].", "See Old Mill.", ("Old Mill",)),
        ("See [[Old Mill|the mill]].", "See the mill.", ("Old Mill",)),
        ("See [[Old Mill#Wheel]].", "See Old Mill > Wheel.", ("Old Mill",)),
        ("See [[Old Mill#Wheel|it]].", "See it.", ("Old Mill",)),
        ("See [[#Wheel]].", "See Wheel.", ()),
        ("See [[Old Mill#^ab12]].", "See Old Mill.", ("Old Mill",)),
        ("| [[Old Mill\\|mill]] |", "| mill |", ("Old Mill",)),
        (
            "See [[Places/Old Mill.md]].",
            "See Places/Old Mill.md.",
            ("Places/Old Mill",),
        ),
        ("Map: ![[map.png]] done", "Map:  done", ()),
        ("![[Old Mill]]", "", ("Old Mill",)),
        ("[[A]] and [[B]] and [[A|again]]", "A and B and again", ("A", "B")),
        ("No links [here].", "No links [here].", ()),
    ],
)
def test_clean_wikilinks(text: str, clean: str, links: tuple[str, ...]) -> None:
    assert clean_wikilinks(text) == (clean, links)


def test_chunks_record_links_and_readable_text() -> None:
    body = "# A\nGo to [[North Gate|the gate]].\n## B\nNo links.\n"
    chunks = chunk_note(make(body))
    assert chunks[0].text == "Go to the gate."
    assert chunks[0].links == ("North Gate",)
    assert chunks[1].links == ()


def test_fixture_note() -> None:
    notes = load_notes(FIXTURE_VAULT, {"scope": ["setting"], "status": ["approved"]})
    harbor = next(n for n in notes.notes if n.path.stem == "example-harbor")
    chunks = chunk_note(harbor)
    assert [c.label for c in chunks] == [
        "example-harbor > Example Harbor",
        "example-harbor > Example Harbor > Ferries",
    ]
    assert chunks[1].text == "Ferries leave for the island at dawn."
    assert chunks[1].links == ("example-island",)
