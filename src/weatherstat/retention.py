"""Retention for control-loop artifacts.

Prunes old command JSON files from the predictions directory and old rows
from the decision log. The collector's EAV snapshots.db is deliberately
never touched — it is the system's training data and is kept forever.

Intended to run at most daily (the runtime loop tracks the last run);
`python -m weatherstat.retention` runs it manually.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from weatherstat.config import (
    COMMAND_RETENTION_DAYS,
    DECISION_LOG_DB,
    DECISION_RETENTION_DAYS,
    PREDICTIONS_DIR,
)

# VACUUM only after bulk deletes: the first run reclaims months of rows,
# but a daily run deletes ~288 and the freelist absorbs those.
_VACUUM_THRESHOLD_ROWS = 1000


def prune_command_files(
    retention_days: int | None = None,
    predictions_dir: Path | None = None,
    now: datetime | None = None,
) -> int:
    """Delete command_*.json files older than the retention window.

    The newest file is always kept regardless of age — the executor reads it.
    Returns the number of files deleted. retention_days <= 0 disables pruning.
    """
    days = COMMAND_RETENTION_DAYS if retention_days is None else retention_days
    if days <= 0:
        return 0
    directory = predictions_dir or PREDICTIONS_DIR
    if not directory.is_dir():
        return 0
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    files = sorted(directory.glob("command_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return 0
    deleted = 0
    for f in files[:-1]:  # always keep the newest
        if f.stat().st_mtime < cutoff_ts:
            f.unlink(missing_ok=True)
            deleted += 1
    return deleted


def prune_decision_log(
    retention_days: int | None = None,
    db_path: Path | None = None,
    now: datetime | None = None,
) -> int:
    """Delete decision rows older than the retention window.

    Returns the number of rows deleted. retention_days <= 0 disables pruning.
    VACUUMs after bulk deletes so the file actually shrinks.
    """
    days = DECISION_RETENTION_DAYS if retention_days is None else retention_days
    if days <= 0:
        return 0
    path = db_path or DECISION_LOG_DB
    if not path.exists():
        return 0
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=days)).isoformat()

    from weatherstat.decision_log import _connect

    conn = _connect(path)
    try:
        cur = conn.execute("DELETE FROM decisions WHERE timestamp < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()
        if deleted >= _VACUUM_THRESHOLD_ROWS:
            conn.execute("VACUUM")
    except sqlite3.OperationalError:
        # Table may not exist yet on a fresh install
        return 0
    finally:
        conn.close()
    return deleted


def run_retention(now: datetime | None = None) -> dict[str, int]:
    """Run all retention tasks; returns per-target deletion counts."""
    return {
        "command_files": prune_command_files(now=now),
        "decision_rows": prune_decision_log(now=now),
    }


def main() -> None:
    counts = run_retention()
    print(
        f"Retention: deleted {counts['command_files']} command file(s) "
        f"(>{COMMAND_RETENTION_DAYS}d), {counts['decision_rows']} decision row(s) "
        f"(>{DECISION_RETENTION_DAYS}d). Snapshots are never pruned."
    )


if __name__ == "__main__":
    main()
