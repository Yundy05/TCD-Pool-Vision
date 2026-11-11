#!/usr/bin/env python3
"""Compare existing detector evidence; this helper is not needed for submission."""

import argparse
import gzip
import hashlib
from pathlib import Path
import sys


def evidence(folder):
    return {
        path.relative_to(folder).as_posix(): path
        for path in folder.rglob("*")
        if path.is_file() and path.name not in {"INDEX.md", ".DS_Store"}
    }


def digest(path):
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Baseline image folder, e.g. <run>/AllBalls")
    parser.add_argument("after", type=Path, help="Revised image folder, e.g. <run>/AllBalls")
    args = parser.parse_args()
    for folder in (args.before, args.after):
        if not folder.is_dir():
            parser.error(f"Not a directory: {folder}")

    before, after = evidence(args.before), evidence(args.after)
    if not before or not after:
        parser.error("Both image folders must contain evidence files")

    missing = sorted(before.keys() - after.keys())
    added = sorted(after.keys() - before.keys())
    shared = before.keys() & after.keys()
    changed = sorted(name for name in shared if digest(before[name]) != digest(after[name]))
    for title, paths in (("Missing", missing), ("Added", added), ("Changed", changed)):
        for path in paths:
            print(f"{title}: {path}")

    passed = not (missing or added or changed)
    print(f"{'PASS' if passed else 'FAIL'}: {len(shared)} shared evidence files; "
          f"{len(missing)} missing, {len(added)} added, {len(changed)} changed")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
