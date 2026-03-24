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
