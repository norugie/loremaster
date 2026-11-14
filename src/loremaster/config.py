"""Load and validate the vault's config file (`lorebot.yaml`) and secrets from env.

Everything vault-specific lives in the config file, kept with the vault. Secrets
(the Discord token) come from environment variables. Both are checked once at
startup; anything missing or invalid raises `ConfigError` with a message naming
the bad value.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

DEFAULT_ANSWER_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_INDEX_PATH = Path("data/index.sqlite")
DEFAULT_DOCUMENT_PREFIX = "search_document: "
DEFAULT_QUERY_PREFIX = "search_query: "
# Top-1 similarity below which nothing counts as relevant.
# Measured with nomic-embed-text on a 40-question set:
# answerable questions scored 0.73-0.86 at the top, unanswerable 0.51-0.66.
DEFAULT_MIN_SCORE = 0.70
# /lore update: best-match score at which a submission counts as "related" to
# an existing note. Measured on 10 test submissions.
DEFAULT_RELATED_MIN_SCORE = 0.73  # lowered from 0.75 after the first measurement
DEFAULT_BACKUP_PATH = Path("data/backups")


class ConfigError(Exception):
    """The config file or environment is missing a value or has an invalid one."""


class _Strict(BaseModel):
    # extra="forbid": a misspelled key is an error instead of being silently ignored.
    # frozen=True: config objects can't be changed after loading.
    model_config = ConfigDict(extra="forbid", frozen=True)


# Discord IDs ("snowflakes") are positive 64-bit integers.
DiscordId = Annotated[int, Field(gt=0)]


class DiscordConfig(_Strict):
    """Discord settings. There are no server IDs: the bot works in any server it
    is invited to. Channel and role IDs are unique across all of Discord, so one
    flat list of each covers every server."""

    # Channels the bot may answer in. Empty means any channel it can see.
    allowed_channel_ids: list[DiscordId] = []
    # Roles allowed to use /lore update. Empty means nobody can.
    staff_role_ids: list[DiscordId] = []


class ModelsConfig(_Strict):
    answer: str = Field(default=DEFAULT_ANSWER_MODEL, min_length=1)
    embedding: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1)
    # Text put before each chunk and each question when embedding them. The
    # defaults are the task prefixes nomic-embed-text expects; other embedding
    # models may need different ones, or "" for none.
    document_prefix: str = DEFAULT_DOCUMENT_PREFIX
    query_prefix: str = DEFAULT_QUERY_PREFIX


class RetrievalConfig(_Strict):
    # Chunks scoring below this (cosine similarity, -1 to 1) are treated as not
    # relevant: they are left out of the prompt, and if none is left the bot
    # says the lore doesn't cover the question. Depends on the embedding model.
    min_score: float = Field(default=DEFAULT_MIN_SCORE, ge=-1, le=1)


class UpdateConfig(_Strict):
    """`/lore update` settings. Off unless `enabled` is true; turning it
    on also needs the vault mounted writable (`VAULT_MODE=rw` in `.env`)."""

    enabled: bool = False
    # A submission whose best search match scores at least this proposes
    # updating that note; below it, a new note.
    related_min_score: float = Field(default=DEFAULT_RELATED_MIN_SCORE, ge=-1, le=1)
    # Folder inside the vault where new notes are written.
    inbox_folder: Path = Path("Inbox")
    # Where originals are copied before an edit. Outside the vault. Relative
    # paths count from the folder the bot runs from, like the default index.
    backup_path: Path = DEFAULT_BACKUP_PATH
    # Frontmatter for new notes. If not set, new notes get the first allowed
    # value of each `include` field, so the bot uses them at once.
    new_note_frontmatter: dict[str, str | list[str]] | None = None


class Config(_Strict):
    """The validated contents of `lorebot.yaml`."""

    vault_path: Path
    # Frontmatter field -> allowed values. A note is included when, for every
    # field listed, its value is one of the allowed values.
    # Each field needs at least one value: an empty list would match no note.
    include: dict[str, Annotated[list[str], Field(min_length=1)]] = Field(min_length=1)
    # Path of the guidance note, relative to the vault. Optional.
    guidance_note: Path | None = None
    # The franchise the setting is based on (e.g. a game series). When set, the
    # bot falls back to the answer model's own knowledge of it if no note is
    # relevant. When not set, it answers from the vault only.
    canon: str | None = Field(default=None, min_length=1)
    discord: DiscordConfig = DiscordConfig()
    models: ModelsConfig = ModelsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    update: UpdateConfig = UpdateConfig()
    # Where the generated search index is stored. Must be outside the vault.
    index_path: Path = DEFAULT_INDEX_PATH


def load_config(path: Path) -> Config:
    """Read, validate and return the config file at `path`.

    A relative `vault_path` (and a relative `index_path` set in the file) is
    resolved against the config file's folder, so a config kept inside its vault
    can say `vault_path: .`. When `index_path` isn't set, the index goes in
    `data/index.sqlite` under the folder the bot runs from. The index must not
    be inside the vault, since the bot never writes there.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"{path}: can't read config file ({e.strerror})") from e

    try:
        data: object = YAML(typ="safe", pure=True).load(text)
    except YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping of settings at the top")

    try:
        config = Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(_describe(path, e)) from e

    base = path.parent
    index_base = base if "index_path" in config.model_fields_set else Path.cwd()
    config = config.model_copy(
        update={
            "vault_path": (base / config.vault_path).resolve(),
            "index_path": (index_base / config.index_path).resolve(),
        }
    )
    if not config.vault_path.is_dir():
        raise ConfigError(f"{path}: vault_path: folder not found: {config.vault_path}")
    if config.index_path.is_relative_to(config.vault_path):
        raise ConfigError(
            f"{path}: index_path: must be outside the vault: {config.index_path}"
        )
    if config.guidance_note is not None:
        guidance = (config.vault_path / config.guidance_note).resolve()
        if not guidance.is_relative_to(config.vault_path):
            raise ConfigError(
                f"{path}: guidance_note: must be inside the vault: {guidance}"
            )
        if not guidance.is_file():
            raise ConfigError(f"{path}: guidance_note: note not found: {guidance}")

    update = config.update
    backup_base = base if "backup_path" in update.model_fields_set else Path.cwd()
    backup = (backup_base / update.backup_path).resolve()
    if backup.is_relative_to(config.vault_path):
        raise ConfigError(
            f"{path}: update.backup_path: must be outside the vault: {backup}"
        )
    inbox = (config.vault_path / update.inbox_folder).resolve()
    if update.inbox_folder.is_absolute() or not inbox.is_relative_to(config.vault_path):
        raise ConfigError(
            f"{path}: update.inbox_folder: must be a folder inside the vault: "
            f"{update.inbox_folder}"
        )
    if any(part.startswith(".") for part in inbox.relative_to(config.vault_path).parts):
        raise ConfigError(
            f"{path}: update.inbox_folder: can't be a hidden (dot) folder, or the "
            "bot would never see the notes it writes there"
        )
    return config.model_copy(
        update={"update": update.model_copy(update={"backup_path": backup})}
    )


def load_discord_token(env: Mapping[str, str]) -> str:
    """Return the Discord bot token from `env` (normally `os.environ`)."""
    token = env.get("DISCORD_TOKEN", "").strip()
    if not token:
        raise ConfigError("DISCORD_TOKEN: not set in the environment or .env")
    return token


def _describe(path: Path, error: ValidationError) -> str:
    """Turn pydantic's errors into one line per bad value, e.g.
    `discord.staff_role_ids[0]: Input should be greater than 0`."""
    lines = [f"{path}: invalid config"]
    for err in error.errors():
        where = ""
        for part in err["loc"]:
            where += f"[{part}]" if isinstance(part, int) else f".{part}"
        lines.append(f"  {where.lstrip('.') or '(top level)'}: {err['msg']}")
    return "\n".join(lines)


def load_discord_guild_id(env: Mapping[str, str]) -> int | None:
    """Return `DISCORD_GUILD_ID` from `env`, or None if it isn't set.

    Development only: when set, slash commands are synced to that one server,
    where they appear at once. When empty, they are synced globally, to every
    server the bot is in (Discord can take up to about an hour to show them).
    """
    raw = env.get("DISCORD_GUILD_ID", "").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) == 0:
        raise ConfigError("DISCORD_GUILD_ID: must be a server ID (digits) or empty")
    return int(raw)
