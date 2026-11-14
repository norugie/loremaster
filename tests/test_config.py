"""Tests for config loading. All configs are made up."""

from pathlib import Path

import pytest

from loremaster.config import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    ConfigError,
    load_config,
    load_discord_guild_id,
    load_discord_token,
)

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"
REPO_ROOT = Path(__file__).parent.parent

VALID = f"""
vault_path: {FIXTURE_VAULT.as_posix()}
include:
  scope: [setting, rules]
  status: [approved]
guidance_note: guidance.md
discord:
  allowed_channel_ids: [201, 202]
  staff_role_ids: [301]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "lorebot.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# Reads vault path, filter, guidance note, channels and staff roles.
def test_reads_all_values(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, VALID))

    assert config.vault_path == FIXTURE_VAULT.resolve()
    assert config.include == {"scope": ["setting", "rules"], "status": ["approved"]}
    assert config.guidance_note == Path("guidance.md")
    assert config.discord.allowed_channel_ids == [201, 202]
    assert config.discord.staff_role_ids == [301]


def test_guidance_note_is_optional(tmp_path: Path) -> None:
    text = VALID.replace("guidance_note: guidance.md\n", "")
    assert load_config(write(tmp_path, text)).guidance_note is None


def test_relative_paths_resolve_from_config_folder(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "guidance.md").write_text("Made-up guidance.", encoding="utf-8")
    text = VALID.replace(FIXTURE_VAULT.as_posix(), "vault")
    config = load_config(write(tmp_path, text + "index_path: cache/i.sqlite\n"))

    assert config.vault_path == vault.resolve()
    assert config.index_path == (tmp_path / "cache" / "i.sqlite").resolve()


def test_default_index_path_is_under_the_run_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    monkeypatch.chdir(run_folder)
    config = load_config(write(tmp_path, VALID))
    assert config.index_path == (run_folder / "data" / "index.sqlite").resolve()


def test_index_path_inside_the_vault_is_refused(tmp_path: Path) -> None:
    (tmp_path / "vault").mkdir()
    text = VALID.replace(FIXTURE_VAULT.as_posix(), "vault")
    text += "index_path: vault/data/index.sqlite\n"
    with pytest.raises(ConfigError, match="index_path: must be outside the vault"):
        load_config(write(tmp_path, text))


def test_embedding_prefixes_default_and_override(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, VALID))
    assert config.models.document_prefix == "search_document: "
    assert config.models.query_prefix == "search_query: "
    text = VALID + "models:\n  document_prefix: ''\n  query_prefix: 'q: '\n"
    config = load_config(write(tmp_path, text))
    assert (config.models.document_prefix, config.models.query_prefix) == ("", "q: ")


# Models default, and the config can override them.
def test_models_default(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, VALID))
    assert config.models.answer == DEFAULT_ANSWER_MODEL
    assert config.models.embedding == DEFAULT_EMBEDDING_MODEL


def test_models_override(tmp_path: Path) -> None:
    text = VALID + "models:\n  answer: some-model:1b\n  embedding: some-embedder\n"
    config = load_config(write(tmp_path, text))
    assert config.models.answer == "some-model:1b"
    assert config.models.embedding == "some-embedder"


# A missing or invalid value stops startup with a message naming it.
@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        (f"vault_path: {FIXTURE_VAULT.as_posix()}\n", "", "vault_path"),
        ("  status: [approved]\n", "  status: approved\n", "include.status"),
        ("  status: [approved]\n", "  status: []\n", "include.status"),
        ("  staff_role_ids: [301]\n", "  staff_role_ids: [-5]\n", "staff_role_ids.0"),
        (
            "  allowed_channel_ids: [201, 202]\n",
            "  allowed_channel_ids: 2\n",
            "allowed",
        ),
        ("  staff_role_ids: [301]\n", "  staff_roles: [301]\n", "discord.staff_roles"),
        ("discord:\n", "discrod:\n", "discrod"),  # misspelled key
    ],
)
def test_missing_or_invalid_value_is_named(
    tmp_path: Path, old: str, new: str, named: str
) -> None:
    assert old in VALID
    with pytest.raises(ConfigError, match=named.replace("[", r"\[")):
        load_config(write(tmp_path, VALID.replace(old, new)))


def test_vault_folder_must_exist(tmp_path: Path) -> None:
    text = VALID.replace(FIXTURE_VAULT.as_posix(), "no-such-folder")
    with pytest.raises(ConfigError, match="vault_path: folder not found"):
        load_config(write(tmp_path, text))


# No server IDs needed; channels and staff roles are optional.
def test_discord_section_is_optional(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, VALID.split("discord:")[0]))
    assert config.discord.allowed_channel_ids == []  # any channel
    assert config.discord.staff_role_ids == []  # nobody may use /lore update


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="can't read config file"):
        load_config(tmp_path / "nope.yaml")


def test_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "include: [unclosed\n"))


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- just\n- a list\n"))


def test_example_config_is_valid_apart_from_its_vault_path(tmp_path: Path) -> None:
    text = (REPO_ROOT / "lorebot.example.yaml").read_text(encoding="utf-8")
    text = text.replace("path/to/your/vault", FIXTURE_VAULT.as_posix())
    load_config(write(tmp_path, text))


# Discord token comes from the environment.
def test_token_from_env() -> None:
    assert load_discord_token({"DISCORD_TOKEN": " abc "}) == "abc"


@pytest.mark.parametrize("env", [{}, {"DISCORD_TOKEN": ""}, {"DISCORD_TOKEN": "  "}])
def test_missing_token(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="DISCORD_TOKEN"):
        load_discord_token(env)


# DISCORD_GUILD_ID is optional and only for development.
@pytest.mark.parametrize(("raw", "expected"), [("", None), ("  ", None), ("123", 123)])
def test_guild_id(raw: str, expected: int | None) -> None:
    assert load_discord_guild_id({"DISCORD_GUILD_ID": raw}) == expected
    assert load_discord_guild_id({}) is None


@pytest.mark.parametrize("raw", ["abc", "-5", "0", "12a"])
def test_bad_guild_id(raw: str) -> None:
    with pytest.raises(ConfigError, match="DISCORD_GUILD_ID"):
        load_discord_guild_id({"DISCORD_GUILD_ID": raw})
