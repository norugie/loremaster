# ARCHITECTURE — Loremaster

Last updated: 2026-11-14. Versions below were checked on 2026-11-12 (sources at the end). Re-check them before upgrading.

## 1. Stack

| Part | Choice | Why |
|---|---|---|
| Language | **Python 3.15** (`requires-python = ">=3.15"`) | The most mature RAG and Discord libraries. Every library below has ready-made 3.15 wheels, so nothing needs a C compiler. |
| Model runtime | **Ollama**, on the Windows host | Free, local, uses the GPU, serves both the chat and the embedding model. |
| Answer model | **`qwen3:4b-instruct-2507-q4_K_M`** (2.5 GB, set in config) | Fits the 3 GB GPU budget. The instruct version has no thinking mode, so replies stay fast. Weaker than the 8B on tricky lore. |
| Embedding model | **nomic-embed-text** (274 MB, set in config) | Small, local, made for search. Runs on the CPU to leave the GPU to the answer model. |
| Ollama client | **`ollama` 0.6.x** (official Python client) | Typed responses. Brings `httpx` and `pydantic`. |
| Discord API | **`discord.py` 2.7.x** | The most used and documented Python Discord library: slash commands, command groups, deferred replies. |
| Vector store | **SQLite** (`sqlite3`, built in) + **`numpy` 2.5.x** | A vault of ~130 notes gives a few thousand 768-number vectors, so brute-force cosine similarity takes milliseconds and no index structure is needed. SQLite also holds the per-note hashes for incremental indexing. |
| Frontmatter | **`ruamel.yaml` 0.19.x** + our own ~20-line splitter | Pure Python. Round-trips YAML keeping key order, comments and formatting, which `/lore update` needs. Headings and wikilinks are parsed by our own code (no Markdown library). |
| File watching | **`watchfiles` 1.3.x** | Async `awatch()` on the same event loop as discord.py, and it batches rapid events. |
| Config | **YAML** (`lorebot.yaml`), validated with **`pydantic` 2.x** | Readable for vault keepers who aren't programmers. Errors name the bad field. `pydantic` already comes with the Ollama client. |
| Tests | **`pytest` 9.x** | The standard Python test runner: plain `assert`, `tmp_path` for made-up temp vaults. |
| Types / lint | **`mypy` 2.4.x** `--strict` + **`ruff` 0.17.x** | `mypy` is the reference type checker. `ruff` lints and formats (like ESLint + Prettier). Both are in the `dev` dependency group (like `devDependencies`). |
| Packaging | **Docker Compose**: `bot` + `app`, with `pyproject.toml` and `uv.lock` | The PC needs only Docker Desktop and Ollama, with no local virtual environment. `uv.lock` keeps installs exact, like `npm ci` with `package-lock.json`. uv is used only inside the Dockerfile. |

**Not used, on purpose:** paid APIs or hosted models, any RAG framework (LangChain, LlamaIndex: RAG is built by hand here), ChromaDB / LanceDB / `sqlite-vec` (more than this size needs), `PyYAML` (no 3.15 wheels, loses formatting on write), `python-frontmatter`, `watchdog` (thread based), `py-cord` (declares `<3.15`), any dotenv library (Compose passes `.env`).

## 2. Data flow

```
Obsidian vault (.md files, mounted read-only at /vault)
  │ at start-up, then on every file change
  ▼
[1] Filter ── parse frontmatter; keep notes matching the config's include filter
  │
  ▼
[2] Chunk ── one chunk per heading section; clean up [[wikilinks]]
  │
  ▼
[3] Embed ── nomic-embed-text: prefix + "note title > heading path" + chunk text
  │
  ▼
[4] Index (SQLite) ── chunk text, vector, note path, heading; a hash per note


Discord: /lore ask <question>
  │
  ▼
[5] Admit ── in a server? allowed channel? server cooldown over?
  │
  ▼
[6] Retrieve ── embed the question; top 5 chunks by cosine similarity
  │
  ├─ best score < retrieval.min_score ──▶ "not found in the lore"
  │                                       (or the wiki fallback, if wiki is set)
  ▼
[7] Prompt ── rules + guidance note, then the chunks (with note names) + question
  │
  ▼
[8] Answer model ── Qwen3 4B Instruct through Ollama
  │
  ▼
Reply: the answer and the names of the notes used
```

Each question stands alone: there is no conversation memory. The vault is only read, except by `/lore update`.

### Why these retrieval choices

- **One chunk per heading.** A section of a lore note is usually one topic. Sections longer than 2,000 characters are split at paragraph, line, then word boundaries, because nomic-embed-text reads at most 2,048 tokens.
- **The label is embedded with the text.** "note title > heading path" in front gives short chunks context.
- **Prefixes.** nomic-embed-text expects `search_document: ` before notes and `search_query: ` before questions. Both are config values, since other embedding models differ.
- **Length-1 vectors.** Vectors are scaled to length 1 when stored, so a plain dot product is the cosine similarity.
- **A score threshold decides "not found".** Below `retrieval.min_score` (default 0.70) no note counts as relevant, and the model isn't called.
- **Small prompt.** The context is 2,048 tokens, because GPU memory grows with it: at most 4,500 characters of notes, answers up to 350 tokens, temperature 0.3.

### Wiki fallback (optional)

Used only when no note is relevant and the config has a `wiki` block. The model never answers from its own knowledge: a 4B model invents canon.

- **Search.** One request to the wiki's MediaWiki API (`generator=search`) returns the three best article titles with their URLs, plus the wiki's name and license.
- **Fetch and clean.** Each page is fetched as rendered HTML (`action=parse`), one request at a time, and turned into plain-text sections by heading with the standard library's `html.parser`. Rendered HTML is used rather than wikitext because templates and infoboxes are already expanded, on any wiki.
- **Rank.** The sections are embedded and scored against the question like vault chunks, with the same `retrieval.min_score` and the same prompt budget. The wiki's own search matches keywords; this step judges meaning.
- **Pages the question names.** The search request also returns each page's redirects. If the question contains a page's title or a redirect to it as whole words, that page's sections go first and skip the score threshold. The embedding model can't know that a nickname or number and the real title mean the same thing; the wiki's redirects do.
- **Answer.** The model gets the relevant sections as "Wiki excerpts" with rules to use only them. The reply says "Not in the server lore" and credits the wiki, its license and the pages used, with links.
- **Nothing relevant, or the wiki fails:** "not found", and the model isn't called.
- **Politeness.** At most four requests per new question, in series, with a 5-second timeout and a 2 MB cap each, gzip, and a User-Agent naming the bot. Searches and pages (with their vectors) are cached in memory for 24 hours, up to 100 each.
- **Trust.** Wiki text is untrusted. It only ever reaches the prompt as excerpts; the bot requests only the configured `api.php`; links are shown only if they point at that wiki.

### Keeping the index current

- **At start-up** only changed notes are re-embedded: new or edited notes (by body hash) are embedded; notes deleted, renamed, moved or no longer passing the filter are removed.
- **While running** the watcher redoes steps [1] to [4] for the notes that changed.
- **A settings change** (embedding model, document prefix, chunk size, chunker version, schema version) empties and rebuilds the index.

### `/lore update` (staff only)

Offered only when `update.enabled` is on and the vault is mounted writable.

- **Find the note.** The submitted lore is searched like a question. A best match at or above `update.related_min_score` (default 0.73) proposes updating that note, otherwise a new note. Staff can switch.
- **Propose.** For an update only the matching section changes. The model's rewrite is kept only if it kept the original lines; otherwise the model writes an addition. A new note goes in `update.inbox_folder`, with `new_note_frontmatter` from the config or frontmatter matching the include filter.
- **Preview.** Only the staff member sees it, with Confirm, switch and Cancel. Nothing is written before Confirm, and it times out after 10 minutes.
- **Apply.** Refused if the note changed since the preview. The original is copied to `update.backup_path` (outside the vault), then a temp dot-file next to the note replaces it in one step. The watcher re-indexes it.

## 3. Folder structure

```
loremaster/
├── CLAUDE.md  ARCHITECTURE.md  README.md
├── pyproject.toml                # metadata, dependencies, tool settings (like package.json)
├── uv.lock                       # exact resolved versions (like package-lock.json)
├── Dockerfile                    # Python 3.15 + locked dependencies; targets runtime and dev
├── compose.yaml                  # services, mounts, Ollama address, polling
├── .dockerignore                 # keeps .env, data and caches out of the image
├── .env.example                  # names only, no values
├── lorebot.example.yaml          # lore-agnostic example config
├── lorebot.yaml                  # local config pointing at a real vault (gitignored)
├── src/
│   └── loremaster/
│       ├── __init__.py           # marks the package; holds __version__
│       ├── config.py             # read + validate lorebot.yaml, fail fast
│       ├── vault.py              # step [1] walk, parse frontmatter, filter
│       ├── chunking.py           # step [2] split by heading, clean wikilinks
│       ├── index.py              # steps [3]+[4] embed, store, sync
│       ├── watcher.py            # re-index on vault file changes
│       ├── retrieve.py           # step [6] cosine similarity search
│       ├── wiki.py               # wiki fallback: search, fetch, clean, rank, cache
│       ├── answer.py             # steps [7]+[8] prompt, model call
│       ├── update.py             # /lore update: propose and write; the only writer
│       └── bot.py                # step [5] and the Discord layer
├── tests/                        # pytest, one test_*.py per module
│   └── fixtures/vault/           # tiny made-up vault (no real lore)
└── data/                         # the index (generated, gitignored)
```

This is the standard Python `src` layout: tests import the installed package rather than whatever files sit in the current folder.

Rule: decisions stay out of the Discord layer. `LoreService` in `bot.py` holds them with no Discord types, so tests run without Discord or Ollama.

## 4. Docker Compose

`compose.yaml` runs two services from one `Dockerfile` (`python:3.15-slim`; uv installs `uv.lock` into the image's own Python):

- **bot**: target `runtime` (453 MB: runtime dependencies only, no uv), `restart: unless-stopped`. Reads `lorebot.yaml` from the repo folder unless `LOREBOT_CONFIG` or `--config` says otherwise, syncs the index before going online, then watches the vault.
- **app**: target `dev` (783 MB: adds `pytest`, `mypy`, `ruff` and uv), for one-off commands.

Both mount the repo at `/app` (live code, and the index in `data/`) and the vault at `/vault`, read-only unless `VAULT_MODE=rw`. `VAULT_PATH` in `.env` picks the vault and defaults to the fixture vault. `.env` is read at run time and never copied into the image. `init: true` lets `docker stop` reach Python.

- **Ollama stays on Windows** with the GPU. The container reaches it at `http://host.docker.internal:11434` (`OLLAMA_HOST`).
- **File watching polls.** Change events from Windows don't cross the bind mount into a Linux container, so Compose sets `WATCHFILES_FORCE_POLLING=true` and `WATCHFILES_POLL_DELAY_MS=300`.

Start: `docker compose up -d bot`. Log: `docker compose logs -f bot`. Stop: `docker compose stop bot`. After code changes: `docker compose restart bot`. Rebuild after changing dependencies: `docker compose build`.

| Command (after `docker compose run --rm app`) | What it does |
|---|---|
| `pytest` · `pytest -m integration` | Unit tests / the tests that call Ollama. |
| `mypy` | Type check in strict mode. |
| `ruff check` · `ruff format` | Lint / format. |
| `python -m loremaster.watcher` | Sync the index once, then watch the vault (no Discord token needed). |
| `python -m loremaster.retrieve "a question"` | Print the top chunks and scores (`-k`, `--text`). |
| `python -m loremaster.answer "a question"` | Print the answer, the notes used and the scores (`--prompt`). |
| `python -m loremaster.wiki "a question"` | Print the wiki sections found and their scores (`-k`, `--text`). |

From Git Bash, prefix commands that pass `/vault` with `MSYS_NO_PATHCONV=1`, or Git Bash rewrites the path. PowerShell doesn't need this.

## 5. Index storage

One SQLite file outside the vault, at `index_path` from the config, defaulting to `data/index.sqlite` under the folder the bot runs from. It is generated: deleting it means a full re-index on the next start. Tables:

| Table | Columns |
|---|---|
| `notes` | `path` PK (vault-relative, forward slashes), `hash` (sha256 of the note's body) |
| `chunks` | `id` PK, `note_path` → `notes` (deleted with its note), `idx` (position in the note), `title`, `headings` (JSON list), `text`, `links` (JSON list), `embedding` (float32, length 1); unique (`note_path`, `idx`) |
| `meta` | `key` PK, `value`: the vector size and a fingerprint of the index settings |

## 6. Discord setup

- **Gateway intents:** none privileged. Slash commands only, and member roles come with each interaction.
- **Scopes:** `bot` + `applications.commands`. **Bot permissions:** View Channels, Send Messages, Send Messages in Threads, Embed Links.
- **Slash commands** sync on start-up: to one test server when `DISCORD_GUILD_ID` is set (instant updates), otherwise globally (Discord can take up to an hour).
- **Which servers:** the config names none, so the bot works in every server it is invited to. Discord IDs are unique across all servers, so `allowed_channel_ids` (empty = any channel) and `staff_role_ids` (empty = nobody) are flat lists that cover them all.
- **Replies** are deferred ("thinking…"), because Discord wants a response within 3 seconds; the answer is edited in. Questions are capped at 300 characters, submitted lore at 1,500.
- **Cooldown:** 15 seconds between questions, per server.

| Command | What it does |
|---|---|
| `/lore ask question:` | Answer from the vault's notes and list the notes used. |
| `/lore update lore:` | Staff only: propose a change to the vault, shown as a private preview to confirm. Not offered when `update.enabled` is off. |

Both are subcommands of one `/lore` command, so Discord's default permissions apply to both; the staff check for `update` is done in code.

## 7. Reliability

- One bad input never stops the bot: a note with broken frontmatter is logged and skipped; a failed question is logged and gets a short error reply, and the next question still works.
- Timeouts: every Ollama call (answer and embedding) gives up after 120 s instead of hanging. Every wiki request gives up after 5 s; a wiki failure becomes a "not found" reply, not an error.
- A note caught half-written or locked keeps its previous version in the index and is retried once after 2 s. Saves are batched (500 ms), so both in-place writes and temp-file-then-rename saves give one sync.
- `/lore update` never half-writes a note: backup first, temp file, then one replace. Any failure leaves the note as it was.
- Config is validated once at start-up and fails fast with a message naming the bad value. With `/lore update` on, the vault must be writable or the bot refuses to start.
- Only one model call runs at a time on the shared GPU, so questions queue.

## 8. Speed and GPU memory

Targets: **median < 3 s, worst < 15 s** from question to answer, and **about 3 GB of GPU memory or less**. Measured on the 4070 with 40 questions (results table in the README):

- **Answers:** median 0.4 s, worst 5.5 s. "Not found" replies take 0.0 s, because the model isn't called.
- **GPU memory:** 2,925 MiB at peak while answering, with the context at 2,048 tokens. nomic-embed-text on the GPU added ~335 MB and went over budget, so it runs on the CPU (`num_gpu: 0`): ~20 ms per question.
- **Indexing:** a full sync of the ~130-note vault takes 0.7 s over the mount. A saved note is searchable 0.25 to 0.64 s later.
- **Idle:** Ollama unloads both models 5 minutes after the last use; the next question reloads them in a few seconds.

## 9. Key trade-offs

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Where vault config lives | With the vault | In the repo | Keeps server-specific details out of the repo; any server can use the bot. |
| Models | Local through Ollama | Hosted APIs | Free and private. |
| Answer model size | 4B Instruct (3 GB budget) | 8B (`qwen3:latest`, 5.2 GB) | The bot sits idle most of the time and shares its GPU with another bot. |
| Thinking mode | None (instruct model) | Thinking model | Faster replies on shared hardware. |
| Note selection | Frontmatter filter in config | Index everything | Only approved notes should be used for answers. |
| RAG framework | None: built by hand | LangChain, LlamaIndex | Every step stays visible, and fewer dependencies. The cost is more code to write and test. |
| Vaults per bot | One vault, any number of servers, none listed in config | One server per bot; listing servers | Several servers can share one setting's vault, and new servers need no config change. |
| Re-indexing | On file change, per note | On a command or restart only | Staff update lore by editing the vault; no admin commands needed. |
| Canon knowledge | A configured wiki, searched when no notes are relevant | The model's own knowledge; canon notes in the vault; a downloaded wiki dump | Settings based on a franchise rely on canon the vault doesn't repeat. The 4B model invented canon when asked from memory. Costs: answers depend on the wiki's keyword search and on the wiki being reachable. |
| Wiki HTTP client | `urllib.request` (standard library) | `httpx`, `aiohttp` | No new dependency for four GET requests; it blocks, but answering already runs in a worker thread. |
| `/lore update` | Edits the vault directly, staff role only | Draft notes for review; posting suggestions to a staff channel | Changes go live at once. Risk: no review; guarded by staff roles, a preview, a backup, and writing only inside the vault. |
| Rate limit | 15 s cooldown, per server | Per member; per channel; none | Protects the shared GPU. |
| Where it runs | Docker container | Natively with uv and a `.venv` | No Python setup on the PC. The costs: file watching has to poll, syncs over the mount are slower, and VS Code needs the Dev Containers extension. |

## Sources (checked 2026-11-12)

- PyPI JSON API (pypi.org/pypi/<name>/json): versions, `requires_python` and Windows wheels for discord.py 2.7.1, aiohttp 3.14.5, ollama 0.6.3, pydantic 2.14.3, numpy 2.5.4, watchfiles 1.3.0 (abi3), ruamel.yaml 0.19.1, pytest 9.2.0, mypy 2.4.1, ruff 0.17.4; and for the rejected py-cord 2.8.1 (`<3.15`), hikari 2.7.0, disnake 2.12.2, sqlite-vec 0.1.9, lancedb 0.41.0, chromadb 1.5.9, watchdog 7.1.0, PyYAML 6.0.3 (no cp315 wheels), python-frontmatter 1.3.0, pyright 1.1.414, ty 0.0.101.
- `uv pip compile --only-binary :all:` (uv 0.13.5) for Python 3.15 on `x86_64-pc-windows-msvc` and `x86_64-manylinux_2_28`: the set resolves with wheels only.
- ollama.com/library: `qwen3:4b-instruct-2507-q4_K_M` 2.5 GB, `qwen3:latest` = `qwen3:8b` 5.2 GB, nomic-embed-text 274 MB (v1.5).
- `ollama show nomic-embed-text`: context length 2048, embedding length 768. Ollama API docs ("Generate Embeddings"): `truncate` defaults to true and cuts off long input; `false` returns an error instead.
- nomic-embed-text model card (huggingface.co/nomic-ai/nomic-embed-text-v1.5, "Task instruction prefixes").
- Installed package sources: `ollama` 0.6.3 (`Client.embed`, `OLLAMA_HOST`), watchfiles 1.3.0 (`awatch`, `Change`, `WATCHFILES_FORCE_POLLING`, `WATCHFILES_POLL_DELAY_MS`).
- docs.discord.com (oauth2, gateway, application commands, receiving and responding): scopes, intents, the 3-second reply rule, the 15-minute interaction token.
- Docker: `python:3.15-slim` checked with `docker manifest inspect` and run (Python 3.15.2, glibc 2.41, SQLite 3.46); container-to-host Ollama checked at `host.docker.internal:11434`; bind-mount events checked on a temp folder (none natively; polling works).
- MediaWiki Action API (checked 2026-11-14): mediawiki.org `API:Etiquette` (requests in series, a descriptive User-Agent, gzip) and `API:Search`; `generator=search` with `prop=info&inprop=url` and `meta=siteinfo`, and `action=parse&prop=text`, tried against a live MediaWiki 1.43 wiki. `prop=extracts` is an extension that not every wiki has, so it isn't used.
- Obsidian save behaviour: forum.obsidian.md/t/atomic-file-writes/56065 (a feature request, not an official statement), plus watchfiles events observed on Windows 11 for in-place writes, temp-file-then-replace saves and renames.
