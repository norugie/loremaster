"""Tests for note filtering. All notes are made up."""

import logging
from pathlib import Path

import pytest

from loremaster.vault import (
    FrontmatterError,
    load_notes,
    matches,
    parse_frontmatter,
    split_frontmatter,
)

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"
INCLUDE = {"scope": ["setting", "rules"], "status": ["approved"]}


def note(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- splitting and parsing ---------------------------------------------------


def test_split_frontmatter() -> None:
    raw, body = split_frontmatter("---\na: 1\n---\n# Title\n")
    assert raw == "a: 1\n"
    assert body == "# Title\n"


def test_split_handles_crlf_and_bom() -> None:
    raw, body = split_frontmatter("﻿---\r\na: 1\r\n---\r\nbody\r\n")
    assert raw == "a: 1\r\n"
    assert body == "body\r\n"


@pytest.mark.parametrize("text", ["# No frontmatter\n", "", "text\n---\na: 1\n---\n"])
def test_no_frontmatter(text: str) -> None:
    assert split_frontmatter(text) == (None, text)


def test_unclosed_frontmatter_is_an_error() -> None:
    with pytest.raises(FrontmatterError, match="not closed"):
        split_frontmatter("---\na: 1\n# Title\n")


def test_empty_frontmatter_counts_as_none() -> None:
    assert parse_frontmatter("---\n---\nbody\n") == (None, "body\n")


@pytest.mark.parametrize("block", ["a: [1\n", "- a\n- b\n", "just text\n"])
def test_bad_frontmatter_raises(block: str) -> None:
    with pytest.raises(FrontmatterError):
        parse_frontmatter(f"---\n{block}---\nbody\n")


# --- the filter rule ------------------------------------------------------------


@pytest.mark.parametrize(
    ("frontmatter", "expected"),
    [
        ({"scope": "setting", "status": "approved"}, True),
        ({"scope": "rules", "status": "approved", "other": 1}, True),
        ({"scope": ["extra", "rules"], "status": "approved"}, True),  # any list item
        ({"scope": "setting", "status": "draft"}, False),  # one field fails: AND
        ({"scope": "Setting", "status": "approved"}, False),  # case-sensitive
        ({"status": "approved"}, False),  # missing field
        ({"scope": [], "status": "approved"}, False),
        ({"scope": None, "status": "approved"}, False),
    ],
)
def test_matches(frontmatter: dict[str, object], expected: bool) -> None:
    assert matches(frontmatter, INCLUDE) is expected


def test_matches_compares_as_text() -> None:
    assert matches({"year": 3, "flag": True}, {"year": ["3"], "flag": ["true"]})


# --- loading a vault ------------------------------------------------------------


def test_fixture_vault() -> None:
    result = load_notes(FIXTURE_VAULT, INCLUDE)

    # Matching notes are kept, with path relative to the vault.
    kept = [n.path.as_posix() for n in result.notes]
    assert kept == ["Places/example-harbor.md", "Places/rules-list.md"]
    assert result.notes[0].frontmatter == {"scope": "setting", "status": "approved"}
    assert result.notes[0].body.lstrip().startswith("# Example Harbor")

    # Everything else is skipped and counted; dot-folders ignored.
    assert result.found == 6
    assert result.skipped_no_frontmatter == 1
    assert result.skipped_not_matching == 2  # draft note, guidance note
    assert result.errors == [Path("Drafts/broken-frontmatter.md")]


def test_bad_note_is_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good = "---\nscope: setting\nstatus: approved\n---\nGood.\n"
    note(tmp_path, "a-bad.md", "---\nscope: [oops\n---\nBad.\n")
    note(tmp_path, "b-good.md", good)
    note(tmp_path, "c-not-utf8.md", "").write_bytes(b"\xff\xfe\x00bad")

    with caplog.at_level(logging.WARNING):
        result = load_notes(tmp_path, INCLUDE)

    assert [n.path.name for n in result.notes] == ["b-good.md"]
    assert [p.name for p in result.errors] == ["a-bad.md", "c-not-utf8.md"]
    assert "a-bad.md" in caplog.text
    assert "c-not-utf8.md" in caplog.text


def test_only_markdown_files(tmp_path: Path) -> None:
    note(tmp_path, "image.png", "not a note")
    note(tmp_path, "data.yaml", "scope: setting\nstatus: approved\n")
    note(tmp_path, "sub/deep/n.md", "---\nscope: rules\nstatus: approved\n---\n")

    result = load_notes(tmp_path, INCLUDE)
    assert [n.path.as_posix() for n in result.notes] == ["sub/deep/n.md"]
    assert result.found == 1


def test_load_does_not_change_the_vault(tmp_path: Path) -> None:
    note(tmp_path, "n.md", "---\nscope: rules\nstatus: approved\n---\nText.\n")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*")}
    load_notes(tmp_path, INCLUDE)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*")}
    assert before == after
