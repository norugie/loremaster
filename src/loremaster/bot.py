"""The Discord bot: `/lore ask` in any server it is invited to.

Two layers:

- `LoreService` holds the bot's decisions with no Discord types in them:
  whether a question may be answered here (DMs, allowed channels), answering
  it off the event loop, and the reply text. Tests drive it directly.
- `LoreBot` is the thin discord.py layer: it registers the `/lore` command
  group, syncs it, defers each reply (Discord wants a response within 3
  seconds; deferring shows "thinking..." and gives us 15 minutes), and sends
  what the service says.

At start-up the index is brought up to date before the bot goes
online, then the vault watcher keeps it current and refreshes the
retriever after each sync. The guidance note is re-read after each sync too,
so edits to it apply without a restart.

Run it (see compose.yaml):

    docker compose up -d bot        # start in the background; shows "online"
    docker compose logs -f bot      # follow its log
    docker compose stop bot         # stop it

The config file is found from `--config`, else the `LOREBOT_CONFIG`
environment variable, else `lorebot.yaml` in the folder it runs from.
"""

import argparse
import asyncio
import io
import logging
import math
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import discord
from discord import app_commands

from loremaster.answer import Answer, Answerer, OllamaChat, load_guidance
from loremaster.chunking import Chunk
from loremaster.config import (
    Config,
    ConfigError,
    load_config,
    load_discord_guild_id,
    load_discord_token,
)
from loremaster.index import IndexSettings, IndexStore, OllamaEmbedder, SyncStats
from loremaster.retrieve import Hit, Retriever
from loremaster.update import Change, UpdateError, Updater
from loremaster.watcher import VaultSyncer, watch_vault

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("lorebot.yaml")
CONFIG_ENV = "LOREBOT_CONFIG"
DISCORD_MESSAGE_LIMIT = 2000
MAX_QUESTION_CHARS = 300
# Seconds between questions in one Discord server.
COOLDOWN_SECONDS = 15.0

DM_REFUSAL = "Lore questions only work inside a server, not in direct messages."
CHANNEL_REFUSAL = "I don't answer lore questions in this channel."
ERROR_REPLY = "Sorry, something went wrong while answering. Please try again later."
UPDATE_OFF = "/lore update is turned off for this bot."
STAFF_ONLY = "Only staff can update the lore."
MAX_LORE_CHARS = 1500
# How long a /lore update preview waits for Confirm or Cancel.
PREVIEW_TIMEOUT_S = 600


# --- pure decisions ----------------------------------------------------------


def channel_allowed(channel_id: int | None, allowed: Sequence[int]) -> bool:
    """An empty list allows every channel."""
    return not allowed or channel_id in allowed


def is_staff(role_ids: Sequence[int], staff_role_ids: Sequence[int]) -> bool:
    """Whether any of the member's roles is a staff role. With no staff roles
    configured nobody is staff."""
    return bool(set(role_ids) & set(staff_role_ids))


def format_reply(
    answer: Answer, question: str = "", limit: int = DISCORD_MESSAGE_LIMIT
) -> str:
    """The question (as a quote), the answer, then the notes it used, within
    Discord's limit.

    The question goes in a `> ` quote line on top, so readers see what was
    asked (it is at most `MAX_QUESTION_CHARS`, on one line). The note list goes
    in Discord's small "subtext" line (`-# `). If the whole message is too long,
    the answer is cut at a word and marked with "…"; the question and note list
    are never cut.
    """
    text = answer.text.strip() or "(The model gave an empty answer.)"
    asked = " ".join(question.split())  # one line, so the quote stays one block
    header = f"> {asked}\n" if asked else ""
    footer = f"\n-# Notes: {', '.join(answer.notes)}" if answer.notes else ""
    room = limit - len(header) - len(footer)
    if len(footer) > limit // 2:  # absurdly many notes: keep the answer readable
        footer = footer[: limit // 2 - 1] + "…"
        room = limit - len(header) - len(footer)
    if len(text) > room:
        cut = text[: room - 1]
        text = (cut.rsplit(" ", 1)[0] if " " in cut else cut) + "…"
    return header + text + footer


def cooldown_message(seconds_left: float) -> str:
    """E.g. "On cooldown: try again in 12 seconds." (rounded up, at least 1)."""
    n = max(1, math.ceil(seconds_left))
    return f"On cooldown: try again in {n} second{'s' if n != 1 else ''}."


class Cooldown:
    """When each Discord server may ask again.

    Keyed by server ID, so each server has its own cooldown, shared by all its
    channels. `clock` is injectable so tests don't have to wait; it defaults to
    `time.monotonic`, which never jumps when the PC's clock is changed.
    """

    def __init__(
        self,
        seconds: float = COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.seconds = seconds
        self._clock = clock
        self._ready_at: dict[int, float] = {}

    def remaining(self, guild_id: int) -> float:
        """Seconds until `guild_id` may ask again; 0 if it may now."""
        return max(0.0, self._ready_at.get(guild_id, 0.0) - self._clock())

    def start(self, guild_id: int) -> None:
        """Begin (or restart) the cooldown for `guild_id` from now."""
        self._ready_at[guild_id] = self._clock() + self.seconds


def find_config_path(arg: Path | None, env: Mapping[str, str]) -> Path:
    """`--config`, else `LOREBOT_CONFIG`, else `lorebot.yaml`."""
    if arg is not None:
        return arg
    from_env = env.get(CONFIG_ENV, "").strip()
    return Path(from_env) if from_env else DEFAULT_CONFIG_PATH


def check_vault_writable(config: Config) -> None:
    """With `/lore update` on, the vault must be writable."""
    if config.update.enabled and not os.access(config.vault_path, os.W_OK):
        raise ConfigError(
            f"update.enabled is true, but the vault is read-only: {config.vault_path}. "
            "Set VAULT_MODE=rw in .env (Docker mounts the vault read-only by "
            "default), or turn update.enabled off."
        )


# --- the bot's logic, without Discord ----------------------------------------


@dataclass(frozen=True)
class Refusal:
    """A reply sent instead of answering, visible only to the asker."""

    message: str


@dataclass(frozen=True)
class Proposal:
    """A `/lore update` change waiting for the staff member's answer."""

    lore: str
    change: Change
    best: Hit | None  # best search match, for the "switch" button

    @property
    def switch_label(self) -> str | None:
        """The switch button's label, or None if there is nothing to switch to."""
        if self.change.kind == "update":
            return "Make a new note instead"
        if self.best is None:
            return None
        return f"Update {self.best.chunk.title} instead"[:80]


def format_preview(
    proposal: Proposal, limit: int = DISCORD_MESSAGE_LIMIT
) -> tuple[str, bool]:
    """The preview message, and whether the text was cut (then the full text is
    attached as a file)."""
    change = proposal.change
    best = proposal.best
    match = f" (best match {best.score:.2f}: {best.chunk.label})" if best else ""
    if change.kind == "update":
        what = (
            "the new text of this section"
            if change.mode == "rewrite"
            else "text to add at the end of this section"
        )
        head = (
            f"**Preview: update {change.title}** › {change.section}\nBelow is {what}."
        )
    else:
        head = (
            f"**Preview: new note** `{change.path.as_posix()}`{match}\n"
            "Below is the note (its frontmatter comes from the config)."
        )
    foot = "\nConfirm to write it, or Cancel. Nothing is written until you confirm."
    room = limit - len(head) - len(foot) - 3  # 3: the newlines around the quote
    body = change.added.strip()
    quoted = "\n".join(f"> {line}" if line else ">" for line in body.splitlines())
    cut = len(quoted) > room
    if cut:
        note = "\n> …(cut; the full text is in the attached file)"
        quoted = quoted[: room - len(note)].rstrip() + note
    return f"{head}\n\n{quoted}\n{foot}", cut


def format_applied(change: Change, written: Path) -> str:
    """The reply after writing: the note's name and what changed."""
    if change.kind == "update":
        how = "rewrote" if change.mode == "rewrite" else "added to"
        return (
            f"Updated **{change.title}**: {how} the section {change.section}. "
            "The original was backed up. The bot will use it within a few seconds."
        )
    return (
        f"Created **{change.title}** at `{written.as_posix()}`. "
        "Sort it into place in Obsidian when you like."
    )


class LoreService:
    """Decides whether to answer and produces reply text. One model call at a
    time, since the GPU is shared."""

    def __init__(
        self,
        config: Config,
        answerer: Answerer,
        cooldown: Cooldown | None = None,
        updater: Updater | None = None,
    ) -> None:
        self.config = config
        self.answerer = answerer
        self.cooldown = cooldown or Cooldown()
        self.updater = updater  # None: /lore update is off
        self._model_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    def admit(self, guild_id: int | None, channel_id: int | None) -> Refusal | None:
        """Why this question can't be answered here, or None if it can.

        Admitting a question starts its server's cooldown at once, before any
        `await`, so two questions arriving together can't both get through.
        """
        if guild_id is None:
            return Refusal(DM_REFUSAL)
        if not channel_allowed(channel_id, self.config.discord.allowed_channel_ids):
            return Refusal(CHANNEL_REFUSAL)
        left = self.cooldown.remaining(guild_id)
        if left > 0:
            return Refusal(cooldown_message(left))
        self.cooldown.start(guild_id)
        return None

    async def ask(self, question: str, guild_id: int | None = None) -> str:
        """Answer one question and return the reply text. Never raises: a
        failure is logged and becomes a short error reply.

        When the answer is done, the server's cooldown starts again, so it
        runs for the full time after the answer.
        """
        try:
            async with self._model_lock:
                answer = await asyncio.to_thread(self.answerer.answer, question)
        except Exception:
            log.exception("answering failed for question %r", question[:80])
            return ERROR_REPLY
        finally:
            if guild_id is not None:
                self.cooldown.start(guild_id)
        log.info(
            "answered in %.1fs from %d notes (%s, top score %s)",
            answer.seconds,
            len(answer.notes),
            "found" if answer.found else "not found",
            f"{answer.top_score:.2f}" if answer.top_score is not None else "-",
        )
        return format_reply(answer, question)

    # --- /lore update ---------------------------------------------

    def admit_update(
        self, guild_id: int | None, role_ids: Sequence[int]
    ) -> Refusal | None:
        """Why this member can't use `/lore update` here, or None if they can."""
        if self.updater is None:
            return Refusal(UPDATE_OFF)
        if guild_id is None:
            return Refusal(DM_REFUSAL)
        if not is_staff(role_ids, self.config.discord.staff_role_ids):
            return Refusal(STAFF_ONLY)  # Nothing is changed
        return None

    async def propose(
        self, lore: str, switch_from: Proposal | None = None
    ) -> Proposal | str:
        """A proposed change for `lore`, or an error message. Writes nothing.

        Normally the target is the best search match if it is related enough,
        else a new note. With `switch_from`, the
        opposite of that earlier proposal: a new note instead of an update, or
        an update of the best match instead of a new note.
        """
        updater = self.updater
        if updater is None:
            return UPDATE_OFF

        def work() -> Proposal:
            target: Chunk | None = None
            if switch_from is None:
                best = updater.best_match(lore)
                if best is not None and updater.is_related(best):
                    target = best.chunk
            else:
                best = switch_from.best
                if switch_from.change.kind == "new" and best is not None:
                    target = best.chunk
            return Proposal(lore, updater.propose(lore, target), best)

        try:
            async with self._model_lock:
                return await asyncio.to_thread(work)
        except UpdateError as e:
            return str(e)
        except Exception:
            log.exception("proposing an update failed")
            return ERROR_REPLY

    async def apply(self, proposal: Proposal) -> str:
        """Write a confirmed change; the reply names the
        note and what changed. Never raises."""
        updater = self.updater
        if updater is None:
            return UPDATE_OFF
        change = proposal.change
        try:
            async with self._write_lock:
                path = await asyncio.to_thread(updater.apply, change)
        except UpdateError as e:
            return f"Nothing was written: {e}"
        except Exception:
            log.exception("writing %s failed", change.path.as_posix())
            return "Nothing was written: something went wrong while saving the note."
        log.info("lore update: %s %s", change.kind, change.path.as_posix())
        return format_applied(change, path.relative_to(updater.vault))

    def reload_guidance(self) -> None:
        """Re-read the guidance note; keep the old text if that fails."""
        try:
            self.answerer.guidance = load_guidance(
                self.config.vault_path, self.config.guidance_note
            )
        except OSError as e:
            log.warning("couldn't re-read the guidance note: %s", e)


# --- Discord -----------------------------------------------------------------


def make_lore_group(service: LoreService) -> app_commands.Group:
    """The `/lore` command group: `ask` always, `update` only when it is on."""
    lore = app_commands.Group(name="lore", description="Questions about the lore")

    @lore.command(name="ask", description="Ask a question about the lore")
    @app_commands.describe(question="Your question")
    async def ask(
        interaction: discord.Interaction,
        question: app_commands.Range[str, 1, MAX_QUESTION_CHARS],
    ) -> None:
        refusal = service.admit(interaction.guild_id, interaction.channel_id)
        if refusal is not None:
            await interaction.response.send_message(refusal.message, ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        reply = await service.ask(question, interaction.guild_id)
        await interaction.followup.send(
            reply, allowed_mentions=discord.AllowedMentions.none()
        )

    if service.updater is None:
        return lore  # /lore update is off: don't offer it

    @lore.command(name="update", description="Staff: add or change lore in the vault")
    @app_commands.describe(lore="The new or changed lore, in plain words")
    async def update(
        interaction: discord.Interaction,
        lore: app_commands.Range[str, 1, MAX_LORE_CHARS],
    ) -> None:
        roles = getattr(interaction.user, "roles", [])  # a Member in a server
        refusal = service.admit_update(interaction.guild_id, [r.id for r in roles])
        if refusal is not None:
            await interaction.response.send_message(refusal.message, ephemeral=True)
            return
        # Ephemeral: only the staff member sees the preview.
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await service.propose(lore)
        if isinstance(result, str):
            await interaction.followup.send(result, ephemeral=True)
            return
        view = UpdateView(service, result, interaction.user.id)
        text, cut = format_preview(result)
        message = await interaction.followup.send(
            text,
            ephemeral=True,
            view=view,
            files=preview_files(result) if cut else [],
            allowed_mentions=discord.AllowedMentions.none(),
            wait=True,
        )
        view.message = message

    return lore


def preview_files(proposal: Proposal) -> list[discord.File]:
    data = proposal.change.added.encode("utf-8")
    return [discord.File(io.BytesIO(data), filename="preview.md")]


class UpdateView(discord.ui.View):
    """Confirm / switch / Cancel buttons under a `/lore update` preview. Only
    the staff member who ran the command can press them; after
    `PREVIEW_TIMEOUT_S` they stop working and nothing is written."""

    def __init__(self, service: LoreService, proposal: Proposal, owner_id: int) -> None:
        super().__init__(timeout=PREVIEW_TIMEOUT_S)
        self.service = service
        self.proposal = proposal
        self.owner_id = owner_id
        self.message: discord.WebhookMessage | None = None
        self.add_item(_ActionButton("confirm", "Confirm", discord.ButtonStyle.success))
        if proposal.switch_label:
            self.add_item(
                _ActionButton(
                    "switch", proposal.switch_label, discord.ButtonStyle.secondary
                )
            )
        self.add_item(_ActionButton("cancel", "Cancel", discord.ButtonStyle.danger))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "Only the person who ran the command can answer this.", ephemeral=True
            )
            return False
        return True

    async def on_timeout(self) -> None:
        if self.message is not None:
            try:
                await self.message.edit(
                    content="Preview timed out: nothing was written.",
                    view=None,
                    attachments=[],
                )
            except discord.HTTPException:
                pass  # the ephemeral message may be gone already

    async def act(self, action: str, interaction: discord.Interaction) -> None:
        if action == "cancel":
            self.stop()
            await interaction.response.edit_message(
                content="Cancelled: nothing was written.", view=None, attachments=[]
            )
            return
        if action == "confirm":
            self.stop()
            await interaction.response.edit_message(
                content="Writing…", view=None, attachments=[]
            )
            reply = await self.service.apply(self.proposal)
            await interaction.edit_original_response(content=reply)
            return
        # switch: propose the other way round, then show a fresh preview
        self.stop()
        await interaction.response.edit_message(
            content="Working on the other option…", view=None, attachments=[]
        )
        result = await self.service.propose(
            self.proposal.lore, switch_from=self.proposal
        )
        if isinstance(result, str):
            await interaction.edit_original_response(content=result)
            return
        view = UpdateView(self.service, result, self.owner_id)
        view.message = self.message
        text, cut = format_preview(result)
        await interaction.edit_original_response(
            content=text,
            view=view,
            attachments=preview_files(result) if cut else [],
        )


class _ActionButton(discord.ui.Button[UpdateView]):
    def __init__(self, action: str, label: str, style: discord.ButtonStyle) -> None:
        super().__init__(label=label, style=style)
        self.action = action

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.view is not None:
            await self.view.act(self.action, interaction)


class LoreBot(discord.Client):
    def __init__(
        self,
        config: Config,
        guild_id: int | None,
    ) -> None:
        # Slash commands need no privileged intents: each interaction carries
        # the member, their roles and the channel. `guilds` keeps the server
        # list current.
        intents = discord.Intents.none()
        intents.guilds = True
        super().__init__(intents=intents)
        self.config = config
        self.guild_id = guild_id
        self.tree = app_commands.CommandTree(self)
        self._tasks: list[asyncio.Task[None]] = []
        self._stores: list[IndexStore] = []

    async def setup_hook(self) -> None:
        """Runs once after login, before the bot connects to the gateway."""
        config = self.config
        embedder = OllamaEmbedder(config.models.embedding)
        settings = IndexSettings(config.models.embedding, config.models.document_prefix)
        service: LoreService | None = None

        def on_sync(_stats: SyncStats) -> None:
            if service is not None:
                service.answerer.retriever.refresh()
                service.reload_guidance()

        syncer = VaultSyncer(
            config.vault_path,
            config.include,
            config.index_path,
            settings,
            embedder,
            on_sync=on_sync,
        )
        # Bring the index up to date before answering anything. Failing here
        # (e.g. Ollama not running) stops the bot with a clear error.
        await asyncio.to_thread(syncer.sync_blocking)
        store = IndexStore(config.index_path, None)
        self._stores.append(store)
        answerer = Answerer(
            Retriever(store, embedder, config.models.query_prefix),
            OllamaChat(config.models.answer),
            load_guidance(config.vault_path, config.guidance_note),
            min_score=config.retrieval.min_score,
            canon=config.canon,
        )
        updater = None
        if config.update.enabled:
            updater = Updater(
                config.vault_path,
                answerer.retriever,
                answerer.model,
                config.update,
                config.include,
            )
            log.info(
                "/lore update is on (staff roles: %d)",
                len(config.discord.staff_role_ids),
            )
        service = LoreService(config, answerer, updater=updater)
        self.tree.add_command(make_lore_group(service))
        self._tasks.append(asyncio.create_task(watch_vault(syncer, initial_sync=False)))
        await self._sync_commands()

    async def _sync_commands(self) -> None:
        if self.guild_id is not None:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info("synced %d command(s) to the test server", len(synced))
        else:
            synced = await self.tree.sync()
            log.info(
                "synced %d command(s) globally; Discord can take up to an hour "
                "to show new or changed commands",
                len(synced),
            )

    async def on_ready(self) -> None:
        log.info("online as %s in %d server(s)", self.user, len(self.guilds))

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for store in self._stores:
            store.close()
        await super().close()


async def run_bot(config: Config, token: str, guild_id: int | None) -> None:
    """Run until `docker stop` (SIGTERM) or Ctrl+C, then log out cleanly."""
    bot = LoreBot(config, guild_id)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.close()))
        except NotImplementedError:  # Windows event loops don't support this
            pass
    async with bot:
        await bot.start(token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m loremaster.bot", description="Run the Loremaster Discord bot."
    )
    parser.add_argument(
        "--config", type=Path, help=f"default: ${CONFIG_ENV} or {DEFAULT_CONFIG_PATH}"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        config = load_config(find_config_path(args.config, os.environ))
        token = load_discord_token(os.environ)
        guild_id = load_discord_guild_id(os.environ)
        check_vault_writable(config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        asyncio.run(run_bot(config, token, guild_id))
    except KeyboardInterrupt:
        pass
    except discord.LoginFailure:
        print("DISCORD_TOKEN was rejected by Discord; check .env", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
