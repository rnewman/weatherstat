"""Benchmark the hourly sysid refit — on a copy, never on the data it is given.

    just bench-sysid refit <data_dir> [--profile out.txt]
        One refit (runtime.sysid_refit, as the TUI/headless loop runs it) on a
        consistent copy of <data_dir>'s snapshots. Stage timings + wall time.

    just bench-sysid periodic <data_dir> [--hold-back-hours 1] [--save params.json]
        The long-running case: in one process, refit on the data minus its
        last hour (cold: full load, empty memos), append that hour, refit
        again (warm: incremental load, tau memo). The warm refit sees exactly
        <data_dir>'s data, so its thermal_params.json (--save) is comparable
        with a from-scratch fit.

    just bench-sysid compare <a.json> <b.json>
        Field-by-field diff of two thermal_params.json (or tracker_state.json)
        files; timestamps ignored. Exit 1 if any number differs by more than
        --rtol (default 1e-9).

<data_dir> is a weatherstat data directory: a synthetic one from
`just synth-db`, or your real ~/.weatherstat — it is only read (via SQLite's
backup API, so a running collector is fine); fits are written to a temporary
copy that is deleted afterwards (--keep to inspect it).
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import json
import os
import pstats
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path


def _workdir(data_dir: Path, keep: bool) -> Path:
    work = Path(tempfile.mkdtemp(prefix="weatherstat-bench-"))
    (work / "snapshots").mkdir()
    shutil.copy2(data_dir / "weatherstat.yaml", work / "weatherstat.yaml")
    if not keep:
        import atexit

        atexit.register(shutil.rmtree, work, ignore_errors=True)
    return work


def _copy_db(src: Path, dst: Path, *, hold_back_rows: int = 0) -> tuple[int, int]:
    """Consistent copy of the readings table; returns (copied, held_back) row counts."""
    with contextlib.closing(sqlite3.connect(src)) as s, contextlib.closing(sqlite3.connect(dst)) as d:
        s.backup(d)
        if hold_back_rows:
            top = d.execute("SELECT MAX(rowid) FROM readings").fetchone()[0]
            d.execute("DELETE FROM readings WHERE rowid > ?", (top - hold_back_rows,))
            d.commit()
        return d.execute("SELECT COUNT(*) FROM readings").fetchone()[0], hold_back_rows


def _refit(label: str, quiet: bool = True) -> float:
    from weatherstat.runtime import sysid_refit

    out = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
        outcome = sysid_refit()
    wall = time.perf_counter() - t0
    stages = next((line.strip() for line in out.getvalue().splitlines() if "Sysid timing" in line), "")
    loaded = next((line.strip() for line in out.getvalue().splitlines() if "snapshots loaded" in line), "")
    print(f"{label:22} {wall:6.1f}s   {loaded}   {stages.removeprefix('Sysid timing: ')}")
    if not outcome.saved:
        print(f"  (not saved: {outcome.message})")
    return wall


def cmd_refit(args: argparse.Namespace) -> None:
    work = _workdir(args.data_dir, args.keep)
    n, _ = _copy_db(args.data_dir / "snapshots" / "snapshots.db", work / "snapshots" / "snapshots.db")
    os.environ["WEATHERSTAT_DATA_DIR"] = str(work)
    print(f"{n:,} readings (copy in {work})")
    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()
    _refit("refit")
    if prof:
        prof.disable()
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(40)
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(25)
        args.profile.write_text(s.getvalue())
        print(f"profile → {args.profile}")


def cmd_periodic(args: argparse.Namespace) -> None:
    src = args.data_dir / "snapshots" / "snapshots.db"
    work = _workdir(args.data_dir, args.keep)
    dst = work / "snapshots" / "snapshots.db"
    with contextlib.closing(sqlite3.connect(src)) as s:
        names = s.execute("SELECT COUNT(DISTINCT name) FROM readings WHERE rowid > (SELECT MAX(rowid) - 5000 FROM readings)").fetchone()[0]
    hold = int(args.hold_back_hours * 12 * names)
    n, _ = _copy_db(src, dst, hold_back_rows=hold)
    os.environ["WEATHERSTAT_DATA_DIR"] = str(work)
    print(f"{n:,} readings, holding back the last {hold:,} (~{args.hold_back_hours:g}h × {names} names); copy in {work}")
    _refit("1. cold process")
    with contextlib.closing(sqlite3.connect(dst)) as d:  # append the hour the way the collector does
        d.execute("ATTACH ? AS src", (str(src),))
        d.execute(
            "INSERT OR IGNORE INTO readings (timestamp, name, value) SELECT timestamp, name, value FROM "
            "(SELECT rowid AS r, timestamp, name, value FROM src.readings ORDER BY rowid DESC LIMIT ?) ORDER BY r",
            (hold,),
        )
        d.commit()
    _refit(f"2. +{args.hold_back_hours:g}h, warm")
    if args.save:
        shutil.copy2(work / "thermal_params.json", args.save)
        print(f"warm thermal_params.json → {args.save}")


def cmd_compare(args: argparse.Namespace) -> None:
    a, b = (json.loads(p.read_text()) for p in (args.a, args.b))
    skip = {"timestamp", "updated_at", "ts"}
    diffs: list[str] = []
    n_numbers = 0

    def walk(x: object, y: object, path: str) -> None:
        nonlocal n_numbers
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                diffs.append(f"{path}: keys differ {sorted(set(x) ^ set(y))}")
            for k in sorted(set(x) & set(y)):
                if k not in skip:
                    walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"{path}: length {len(x)} vs {len(y)}")
                return
            for i, (u, v) in enumerate(zip(x, y, strict=True)):
                walk(u, v, f"{path}[{i}]")
        elif isinstance(x, int | float) and isinstance(y, int | float) and not isinstance(x, bool):
            n_numbers += 1
            if x != y and abs(x - y) > args.rtol * max(abs(x), abs(y)):
                diffs.append(f"{path}: {x!r} vs {y!r}")
        elif x != y:
            diffs.append(f"{path}: {x!r} vs {y!r}")

    walk(a, b, "")
    print(f"{n_numbers} numbers compared; {len(diffs)} differences (rtol {args.rtol:g})")
    for d in diffs[:30]:
        print("  ", d)
    sys.exit(1 if diffs else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("refit", help="one refit on a copy")
    p.add_argument("data_dir", type=Path)
    p.add_argument("--profile", type=Path, help="write a cProfile report here")
    p.add_argument("--keep", action="store_true", help="keep the temporary copy")
    p.set_defaults(fn=cmd_refit)
    p = sub.add_parser("periodic", help="cold refit, append, warm refit — one process")
    p.add_argument("data_dir", type=Path)
    p.add_argument("--hold-back-hours", type=float, default=1.0)
    p.add_argument("--save", type=Path, help="copy the warm thermal_params.json here")
    p.add_argument("--keep", action="store_true", help="keep the temporary copy")
    p.set_defaults(fn=cmd_periodic)
    p = sub.add_parser("compare", help="diff two thermal_params.json files")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.set_defaults(fn=cmd_compare)
    args = parser.parse_args()
    if hasattr(args, "data_dir"):
        args.data_dir = args.data_dir.expanduser().resolve()
        if not (args.data_dir / "snapshots" / "snapshots.db").exists():
            sys.exit(f"no snapshots/snapshots.db under {args.data_dir}")
    args.fn(args)


if __name__ == "__main__":
    main()
