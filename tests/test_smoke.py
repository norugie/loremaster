"""Smoke test: the package installs and imports."""

import loremaster


def test_package_has_version() -> None:
    assert loremaster.__version__ == "0.1.0"
