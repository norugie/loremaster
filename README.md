# Loremaster

A self-hosted Discord bot that answers lore questions from your roleplay server's Obsidian vault, and names the notes each answer came from. If the notes don't cover a question, it says so instead of guessing.

It runs on your own PC: the bot (Python, discord.py) in a Docker container, and two small local models through [Ollama](https://ollama.com), one for search and one for writing answers. Everything is free and local, and nothing about your vault goes anywhere else. One running bot serves one vault, in as many Discord servers as you invite it to.

## Requirements

**Software**

- **Docker Desktop** ([docs.docker.com](https://docs.docker.com/desktop/)). Tested on Windows with the WSL2 backend.
- **Ollama** installed on the same PC, outside Docker. It runs the models on your graphics card.
- **Git**, to clone the repo
- **An Obsidian vault** with your lore, on the same PC
- For development: nothing more. Tests and tools run in a Docker container too.

**Hardware**

| | Minimum | Notes |
|---|---|---|
| GPU | Any card Ollama supports | Tested on an RTX 4070. |
| VRAM | about 3 GB free | Measured 2.9 GB while answering. Only the answer model uses the GPU; the search model runs on the processor. |
| Disk | about 3 GB for the models, plus the Docker image (estimate) | `qwen3:4b-instruct-2507-q4_K_M` is 2.5 GB and `nomic-embed-text` is about 0.3 GB. The search index is small. |
| Network | Internet for Discord | The bot only makes outgoing connections; no ports need to be opened. |

## Install

1. **Create the Discord bot.** In the [Developer Portal](https://discord.com/developers/applications), make a new application. The name is the bot's name.
   - **Bot** tab: choose **Reset Token** and copy the token. Keep it secret: it is the bot's password. Leave every **Privileged Gateway Intent** off; the bot only uses slash commands.
   - **General Information**: copy the Application ID.
2. **Invite it** to your server (replace `CLIENT_ID`):
   ```
   https://discord.com/oauth2/authorize?client_id=CLIENT_ID&scope=bot+applications.commands&permissions=274877926400
   ```
   This gives View Channels, Send Messages, Send Messages in Threads and Embed Links.
3. **Clone and pull the models.**
   ```powershell
   git clone https://github.com/norugie/loremaster.git
   cd loremaster
   ollama pull qwen3:4b-instruct-2507-q4_K_M
   ollama pull nomic-embed-text
   ```
4. **Configure.**
   ```powershell
   copy .env.example .env
   copy lorebot.example.yaml lorebot.yaml
   ```
   Fill in `.env`. It holds secrets and settings for this PC; it is never committed or copied into the Docker image.

   | Setting | What it is |
   |---|---|
   | `DISCORD_TOKEN` | The bot token from step 1. Required. |
   | `DISCORD_CLIENT_ID` | The Application ID. Optional; handy for building the invite link. |
   | `DISCORD_GUILD_ID` | Optional, for testing. A server ID: commands show up there at once. Leave it empty and commands are registered for every server, which Discord can take up to an hour to show. |
   | `VAULT_PATH` | The folder of your vault on this PC, e.g. `C:\Users\you\Documents\Obsidian\MyLore`. Docker mounts it read-only at `/vault`. |
   | `VAULT_MODE` | Optional. `rw` mounts the vault writable, which only `/lore update` needs. Default `ro`: read-only. |
   | `LOREBOT_CONFIG` | Optional. Where the config file is, as the bot sees it. Default: `lorebot.yaml` in the repo folder. |
   | `OLLAMA_HOST` | Optional. Where Ollama runs. Default: the Windows/Mac host, `http://host.docker.internal:11434`. |

   Then edit `lorebot.yaml` (see [Configuration](#configuration)). You can keep it in the repo folder, or inside your vault so it travels with the lore: then set `LOREBOT_CONFIG=/vault/lorebot.yaml` in `.env` and write `vault_path: .` in it.
5. **Start it.**
   ```powershell
   docker compose up -d --build bot
   ```
   The first start indexes your vault, which takes a few seconds to a few minutes depending on its size. When `docker compose logs -f bot` says `online as YourBot#1234`, the bot shows as online in Discord.

Stop with `docker compose stop bot`. Restart with `docker compose restart bot`, for example after changing the config. Logs: `docker compose logs -f bot`. The bot starts again by itself when Docker Desktop starts, until you stop it. The search index is kept in `data/index.sqlite`.

## Using it

Type `/lore ask` and your question in any channel the bot can see. The reply is the answer, then a small line listing the notes it used.

| Command | What it does |
|---|---|
| `/lore ask question:` | Ask anything about the lore. Each question is answered on its own: the bot doesn't remember earlier ones. |
| `/lore update lore:` | Staff only, off by default. Add or change lore in the vault from Discord, after a preview you confirm. |

- If no note is close enough to the question, the bot says it couldn't find it in the lore. If you set a `wiki` in the config, it first searches that wiki and answers from the pages it finds, marked "Not in the server lore" with links to them. It never answers from the model's own memory.
- After a question, the same Discord server has to wait 15 seconds before the next one. Whoever asks too soon sees how long is left (only they see it). Other servers aren't affected.
- The bot doesn't answer in direct messages, or in channels left off `allowed_channel_ids` if you set it.
- Edit the vault in Obsidian as usual. The bot notices saved changes within a second or so and uses them for the next question; no restart needed.

### `/lore update`

This is the only thing that ever writes to your vault. To turn it on:

1. In the config, set `update: { enabled: true }` and list your staff roles in `discord.staff_role_ids` (with none listed, nobody can use it).
2. In `.env`, set `VAULT_MODE=rw` so Docker mounts the vault writable. (If you forget, the bot refuses to start and says so.)
3. `docker compose up -d bot` to restart it with the new mount.

What happens when staff use it:

- The bot searches for the note the lore belongs to. If one matches well enough
  (`update.related_min_score`), it proposes changing that note; otherwise it
  proposes a new note in the inbox folder.
- **Only the matching section** (heading) of an existing note is changed; the
  frontmatter and the rest of the note stay exactly as they were. The small
  model is only trusted to rewrite a section when its rewrite keeps the original
  lines (so a single fact can be corrected); otherwise it adds the new lore at
  the end of the section instead.
- **You see a preview first**, visible only to you, with **Confirm**, **Cancel**,
  and a button to switch between "update that note" and "make a new note".
  Nothing is written until you confirm; after 10 minutes the preview expires.
- Before an existing note is changed, the original is copied to the backup
  folder (`update.backup_path`, outside the vault), so you can always undo by
  hand. If someone edited the note since the preview, nothing is written.
- New notes get the frontmatter from `update.new_note_frontmatter`. Set it to
  something your `include` filter doesn't match (e.g. `status: draft`) to review
  new notes before the bot answers from them.

## Configuration

`lorebot.yaml` describes your vault. Only `vault_path` and `include` are
required. A made-up example with every field:

```yaml
vault_path: /vault                 # the vault, as Docker mounts it

include:                           # which notes the bot may use
  scope: [setting, rules]
  status: [approved]

guidance_note: Meta/bot-guidance.md

wiki:
  api_url: https://starfall.example.org/api.php

canon: The Starfall Chronicles

discord:
  allowed_channel_ids: [222222222222222222]
  staff_role_ids: [333333333333333333]

models:
  answer: qwen3:4b-instruct-2507-q4_K_M
  embedding: nomic-embed-text
  document_prefix: "search_document: "
  query_prefix: "search_query: "

retrieval:
  min_score: 0.70

update:
  enabled: true
  related_min_score: 0.73
  inbox_folder: Inbox
  backup_path: data/backups
  new_note_frontmatter:
    status: draft

index_path: data/index.sqlite
```

| Field | Required | What it does |
|---|---|---|
| `vault_path` | yes | Folder of the vault. With Docker this is `/vault` (or `.` when the config file sits in the vault). Relative paths count from the config file's folder. |
| `include` | yes | Which notes the bot may read, by frontmatter. Each key is a frontmatter field and lists the allowed values. A note is used only if **every** listed field matches; when the note's value is a list (like tags), one allowed item is enough. Values are compared exactly (case matters). Notes without frontmatter are skipped, as are folders and files starting with `.`. |
| `guidance_note` | no | A note (path inside the vault) added to every answer prompt: house rules for answering, such as "server lore overrides canon" or "call the city by its full name". Keep it short. The bot won't start if the note doesn't exist. Edits apply without a restart. |
| `wiki.api_url` | no | A wiki to search when no note is relevant: the full `https://` address of its `api.php` (any MediaWiki wiki, which most fan wikis are). The bot reads the three best-matching pages, answers only from their text, and credits the wiki with links. If the wiki has nothing relevant or can't be reached, the bot says it couldn't find it. Your notes always win when they cover the question. Each new question costs the wiki up to four requests; pages are remembered for a day. Check that the wiki's terms allow automated requests before you turn this on. |
| `canon` | no | The franchise your setting is based on. Only used to tell the model what the wiki is about; it does nothing without `wiki`. |
| `discord.allowed_channel_ids` | no | Channels where the bot answers. Empty or missing: any channel it can see. Channel IDs are unique across Discord, so one list covers every server. |
| `discord.staff_role_ids` | no | Roles allowed to use `/lore update`. Empty or missing: nobody. |
| `models.answer` | no | The Ollama model that writes answers. Default `qwen3:4b-instruct-2507-q4_K_M`. |
| `models.embedding` | no | The Ollama model used for search. Default `nomic-embed-text`. Changing it rebuilds the index. |
| `models.document_prefix`, `models.query_prefix` | no | Text put before notes and questions when embedding them. The defaults are what nomic-embed-text expects; use `""` for a model that needs none. |
| `retrieval.min_score` | no | How similar (0 to 1) a note must be to the question to count as relevant. Default 0.70, which suits nomic-embed-text. Lower it if the bot says "not found" for things the vault does cover; raise it if it answers from unrelated notes. |
| `update.enabled` | no | Turns on `/lore update` (default off). Also needs `VAULT_MODE=rw` in `.env`. |
| `update.related_min_score` | no | How well the lore must match an existing note's section (0 to 1) for the bot to propose updating that note rather than making a new one. Default 0.73. |
| `update.inbox_folder` | no | Folder inside the vault for new notes. Default `Inbox`. |
| `update.backup_path` | no | Where originals are copied before an edit. Must be outside the vault. Default `data/backups` in the folder the bot runs from. |
| `update.new_note_frontmatter` | no | Frontmatter for new notes, e.g. `status: draft`. If not set, new notes get frontmatter matching `include`, so the bot uses them at once. |
| `index_path` | no | Where the search index is stored. Default `data/index.sqlite` in the folder the bot runs from. Must be outside the vault. It is generated: deleting it just means a full re-index on the next start. |

To copy a Discord ID, turn on **Developer Mode** (User Settings → Advanced), then
right-click a channel or role and choose **Copy ID**.

## Your vault

The bot reads only the notes your `include` filter allows, splits each into
sections by heading, and searches those sections. These habits help it find the
right ones and cite them clearly:

- **Frontmatter on every lore note.** Give notes the fields your filter checks,
  for example:

  ```yaml
  ---
  scope: setting
  status: approved
  ---
  ```

  Drafts, personal notes and out-of-character notes stay out simply by not
  matching. A note whose frontmatter is broken is skipped and logged; fix it and
  it's picked up on save.
- **Clear note titles.** Answers cite notes by file name (as Obsidian shows it),
  so `harbor-district.md` reads better than `notes-3.md`. Keep file names unique.
- **Headings for each topic.** Each heading starts a new section, and the
  heading path (`Note > Heading > Subheading`) is part of what gets searched. A
  note with one topic per heading is found more precisely than one long wall of
  text. Very long sections are split further automatically.
- **Say things plainly in each section.** A section should make sense on its
  own: name the place or person instead of relying on "she" or "it" from three
  sections earlier.
- **Links are fine.** `[[wikilinks]]` are turned into readable text. Sections
  that are only lists of links (like a "Navigation" section) carry no facts,
  though, and can crowd out useful ones.
- **Keep the guidance note short**, since it goes into every prompt.

## Troubleshooting

- **`/lore` doesn't show up in Discord.** With `DISCORD_GUILD_ID` empty, Discord
  can take up to an hour to show new or changed commands. To test right away,
  put your server's ID in `DISCORD_GUILD_ID` and restart the bot. Also check the
  bot was installed with the `applications.commands` scope.
- **The bot never comes online** (`docker compose ps` shows it "Restarting").
  Docker keeps restarting a bot that stops on an error, so run
  `docker compose logs bot` to see why. A config problem names the bad field
  (e.g. `include: Field required`); `DISCORD_TOKEN: not set` means `.env` has no
  token; `DISCORD_TOKEN was rejected` means the token is wrong or was reset. Fix
  it, then `docker compose up -d bot` again.
- **"Something went wrong while answering."** Usually Ollama isn't running or a
  model isn't pulled. Check `ollama list` shows both models, then ask again.
- **The first answer after a while is slow.** Ollama unloads models after 5
  minutes idle and reloads them on the next question (a few seconds).
- **Answers miss notes you expected.** Check the note matches `include`, then try
  the search by hand (below) to see what the bot finds and how it scores.

## Development

Everything runs in the `app` container; nothing but Docker and Ollama is needed on the PC.

```powershell
docker compose run --rm app pytest                  # unit tests
docker compose run --rm app pytest -m integration   # integration tests (need Ollama running)
docker compose run --rm app mypy                    # type check (strict)
docker compose run --rm app ruff check              # lint
docker compose run --rm app python -m loremaster.retrieve "a question"   # see what search finds
docker compose run --rm app python -m loremaster.answer "a question"     # answer without Discord
docker compose run --rm app python -m loremaster.wiki "a question"       # see what the wiki search finds
docker compose run --rm app python -m loremaster.watcher                 # build or refresh the index
```

The last four read the same `lorebot.yaml` as the bot. The watcher builds the index without starting the bot (no Discord token needed): press Ctrl+C after its first `index sync` line.

How it's built: see [ARCHITECTURE.md](ARCHITECTURE.md).

## Results

Each measurement had a target set beforehand. Measured on 12 November 2026 on an RTX 4070, with 40 questions (30 the vault can answer, 10 it can't) written against a private vault of about 130 notes. The question sets contain vault text, so they stay out of the repo.

| Eval | Target | Measured |
|---|---|---|
| Search finds the right note (hit@5 / hit@1 / MRR) | 0.90 / 0.80 / 0.85 or better | 0.97 / 0.90 / 0.94 |
| Answers contain only what the notes say | 0.90 or better | Judged by hand: the 30 answers were all okay |
| The expected note is listed under the answer | 0.90 or better | 1.00 |
| Says "not found" when the vault can't answer | 0.90 or better | 1.00 |
| Says "not found" by mistake | 0.10 or less | 0.03 |
| Time to answer, median / worst | under 3 s / under 15 s | 0.4 s / 5.5 s |
| GPU memory while answering | about 3 GB or less | 2,925 MiB |
| `/lore update`: right note / new lore included / rest of the note intact | 0.90 / 0.90 / 1.00 or better | 0.90 / 1.00 / 1.00 (10 submissions) |

- Questions with no relevant note are answered in 0.0 s, because the model isn't called. That pulls the median down.
- Server-over-canon answers (target 0.80) have no questions yet. The wiki fallback was added after these measurements and isn't covered by them.

How it was measured: eval scripts ran the question files through the real search and models; GPU memory was sampled with `nvidia-smi` every 200 ms during a full run. The scripts were removed once the targets were met, so they are not in this repo.
