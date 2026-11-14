"""Loremaster: answers lore questions from an Obsidian vault, citing the notes used."""

from importlib.metadata import version

# Read from the installed package metadata, so pyproject.toml is the
# one place where the version is set.
__version__: str = version("loremaster")
