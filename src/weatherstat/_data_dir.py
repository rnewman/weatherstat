"""Shared data directory resolution. No dependencies on other weatherstat modules."""

import os
from pathlib import Path


def resolve_data_dir() -> Path:
    """Resolve the weatherstat data directory.

    Priority: WEATHERSTAT_DATA_DIR env var > ~/.weatherstat
    """
    env = os.environ.get("WEATHERSTAT_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".weatherstat"


def load_env_file(path: Path) -> list[str]:
    """Source a simple KEY=VALUE file into os.environ for vars not already set.

    Returns the list of keys that were loaded. Silently no-ops if the file is
    missing. Strips surrounding quotes and ignores blank lines and `#` comments.
    """
    if not path.is_file():
        return []
    loaded: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded
