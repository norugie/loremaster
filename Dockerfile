# Loremaster images: Python 3.15 with the locked dependencies installed.
# uv is used only at build time to install the exact versions in uv.lock;
# nothing is installed on the host and there is no local virtual environment.
#
# Two targets (compose.yaml picks one per service):
#   runtime  the bot: runtime dependencies only, no uv, no test tools
#   dev      everything for tests and tools: adds pytest, mypy, ruff and uv
FROM python:3.15-slim AS runtime

# Install into the image's own Python (no venv inside the container either),
# precompile .pyc files for faster start-up, and copy instead of hard-linking.
ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# uv is mounted only for these RUN steps, so it isn't left in the image.
# Dependencies first, in their own layer, so code edits don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.5,source=/uv,target=/bin/uv \
    uv sync --locked --no-dev --no-install-project

# Then the package itself (installed editable, so a mounted /app/src is used).
COPY README.md ./
COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.5,source=/uv,target=/bin/uv \
    uv sync --locked --no-dev

CMD ["python", "-m", "loremaster.bot"]


FROM runtime AS dev

# uv stays in this image so `docker compose run --rm app uv add ...` works.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
RUN uv sync --locked --all-groups

CMD ["python", "-m", "pytest"]
