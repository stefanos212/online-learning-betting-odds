"""
SNAPSHOT results/ AND COMPARE IT AFTER A CHANGE.

The project's rule for touching anything shared is: back up the outputs, rerun,
and prove the outputs are unchanged. This automates that rule so it can be run
after every refactoring step instead of by hand.

CSVs are compared on values rather than bytes, because row order can legitimately
differ (dict iteration, ties in an unstable sort) without any number moving. PNGs
are compared by hash, since a figure that is value-identical is also byte-identical
here: every plot is written by the same matplotlib version from the same data.

Usage:
  python check_results.py save BASELINE      copy results/ to BASELINE
  python check_results.py diff BASELINE      compare current results/ to it
"""

import filecmp
import hashlib
import os
import shutil
import sys

import pandas as pd

RESULTS = os.path.join("..", "results")
RTOL = 1e-9


def files(d):
    return sorted(f for f in os.listdir(d)
                  if os.path.isfile(os.path.join(d, f)))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(dest):
    if os.path.exists(dest):
        sys.exit(f"{dest} already exists; remove it or pick another name")
    shutil.copytree(RESULTS, dest)
    n = len(files(dest))
    print(f"saved {n} files to {dest}")


def compare_csv(a, b):
    """Return None if equal, else a short description of the first difference."""
    da, db = pd.read_csv(a), pd.read_csv(b)
    if list(da.columns) != list(db.columns):
        only_a = set(da.columns) - set(db.columns)
        only_b = set(db.columns) - set(da.columns)
        if only_a or only_b:
            return f"columns differ (+{sorted(only_b)} -{sorted(only_a)})"
        return "column order differs"
    if len(da) != len(db):
        return f"{len(da)} rows -> {len(db)} rows"
    key = [c for c in da.columns if da[c].dtype == object]
    if key:
        da = da.sort_values(key, kind="mergesort").reset_index(drop=True)
        db = db.sort_values(key, kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(da, db, check_exact=False,
                                      rtol=RTOL, atol=0, check_dtype=False)
    except AssertionError as e:
        return str(e).split("\n")[0][:160]
    return None


def diff(base):
    if not os.path.isdir(base):
        sys.exit(f"no such baseline: {base}")
    fa, fb = set(files(base)), set(files(RESULTS))
    removed, added = sorted(fa - fb), sorted(fb - fa)
    same, changed = [], []

    for f in sorted(fa & fb):
        pa, pb = os.path.join(base, f), os.path.join(RESULTS, f)
        if f.endswith(".csv"):
            d = compare_csv(pa, pb)
            (same if d is None else changed).append((f, d))
        else:
            ok = filecmp.cmp(pa, pb, shallow=False)
            (same if ok else changed).append((f, None if ok else "bytes differ"))

    print(f"baseline {base}: {len(fa)} files, current: {len(fb)} files\n")
    print(f"  identical : {len(same)}")
    print(f"  changed   : {len(changed)}")
    print(f"  added     : {len(added)}")
    print(f"  removed   : {len(removed)}")
    for f in added:
        print(f"    + {f}")
    for f in removed:
        print(f"    - {f}")
    for f, d in changed:
        print(f"    ~ {f}: {d}")
    ok = not changed and not removed
    print("\n" + ("UNCHANGED" if ok else "DIFFERENCES FOUND"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("save", "diff"):
        sys.exit(__doc__)
    (save if sys.argv[1] == "save" else diff)(sys.argv[2])
