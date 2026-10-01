#!/usr/bin/env python
"""Delete unreadable (truncated) parquet files left behind by a crashed collection run.

A collection process killed while writing (e.g. Ray OOM) leaves a half-written
``rank_*/id_*/data/chunk-*/file-*.parquet``. ``convert_rlinf_collected_to_v21.py`` reads the data files
directly (not the meta), so deleting the broken file keeps the remaining episodes consistent.

Safety:
  * only files that fail to open as parquet are touched (a complete file is never deleted);
  * a file modified less than --min-age-min minutes ago is skipped, because a running collection
    may still be writing it. Run this after the collection process has ended.
  * --dry-run only lists what would be deleted.

Usage:
    python scripts/remove_corrupt_parquets.py collected_v2/smoke/m0 collected_v2/smoke/m1 ...
    python scripts/remove_corrupt_parquets.py collected_v2          # every shard below a directory
    python scripts/remove_corrupt_parquets.py --dry-run collected_v2
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq


def is_readable(path: Path) -> tuple[bool, str]:
    try:
        pf = pq.ParquetFile(path)
        pf.read(columns=["episode_index"])  # a truncated file fails at open or on the first column read
        return True, ""
    except Exception as e:  # noqa: BLE001 - any failure means the file is unusable
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+", help="collection directory (searched recursively for data/chunk-*/*.parquet)")
    ap.add_argument("--dry-run", action="store_true", help="list only, delete nothing")
    ap.add_argument("--min-age-min", type=float, default=10.0,
                    help="skip files modified more recently than this (a running collection may still write them)")
    args = ap.parse_args()

    n_total = n_bad = n_deleted = n_young = 0
    now = time.time()
    for s in args.src:
        files = sorted(glob.glob(str(Path(s) / "**" / "data" / "chunk-*" / "*.parquet"), recursive=True))
        for f in files:
            n_total += 1
            ok, why = is_readable(Path(f))
            if ok:
                continue
            n_bad += 1
            age_min = (now - Path(f).stat().st_mtime) / 60
            size_mb = Path(f).stat().st_size / 1e6
            if age_min < args.min_age_min:
                n_young += 1
                print(f"[SKIP young {age_min:.1f} min < {args.min_age_min:g}] {f} ({size_mb:.1f} MB, {why})")
                continue
            if args.dry_run:
                print(f"[WOULD DELETE] {f} ({size_mb:.1f} MB, {why})")
            else:
                Path(f).unlink()
                n_deleted += 1
                print(f"[DELETED] {f} ({size_mb:.1f} MB, {why})")
    print(f"\nchecked {n_total} parquet file(s): {n_bad} unreadable, {n_deleted} deleted, {n_young} skipped as too recent"
          + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
