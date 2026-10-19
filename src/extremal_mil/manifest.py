"""A record of every sweep cell that has been computed.

Each entry holds the file's name, its full parameters (read from inside the
file, not parsed out of its name), how long it took, how large its dataset
was, and the git commit that produced it. That is enough to answer the three
questions the filenames cannot:

- what do we have, and at what parameters?
- what would it cost to regenerate?
- which code produced it, and was the tree clean at the time?

Generators call :func:`record` after writing a cell. Everything already on
disk is picked up by :func:`backfill`, which reads each file's own stored
metadata, so no information is invented.

Usage
-----
::

    python -m extremal_mil.manifest summary
    python -m extremal_mil.manifest backfill [--log LOG ...]

``backfill`` is idempotent:  a file already in the manifest is left alone.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.jsonl"

# The data directories this project writes cells into.
DATA_DIRS = [
    Path(__file__).resolve().parents[2] / "notebooks" / "fig2_data",
    Path(__file__).resolve().parents[2] / "notebooks" / "fig4_data",
    Path(__file__).resolve().parents[2] / "notebooks" / "si_data",
]

# Which family a file belongs to, by filename prefix. The families differ in
# what a "cell" means, not just in naming: see each generator's docstring.
FAMILIES = [
    ("trajectory", re.compile(r"^traj_\w+?_l\d+_d\d+_lr[\d.]+_s\d+\.npz$")),
    ("trajectory_legacy", re.compile(r"^traj_d\d+_l\d+\.npz$")),
    ("gradient_curves", re.compile(r"^curves_lam[\d.]+_l\d+_d\d+\.npz$")),
    ("endpoint_sweep", re.compile(r"^sweep_\w+?_l\d+_d\d+\.npz$")),
    ("si_gradient_curves", re.compile(r"^si_u_a[\d.]+_lam[\d.]+_l\d+_d\d+\.npz$")),
    ("si_lambda_sweep", re.compile(r"^si_lam_u[\d.]+_l\d+_d\d+\.npz$")),
    ("si_noise", re.compile(r"^si_noise_l\d+_d\d+\.npz$")),
    ("si_gradient_convergence",
     re.compile(r"^gradconv_l\d+_lam[\d.]+_d\d+\.npz$")),
    ("classification_trajectory",
     re.compile(r"^cls_traj_l\d+_d\d+_lr[\dp.]+_s\d+\.npz$")),
    ("classification_curves",
     re.compile(r"^cls_curves_lam[\d.]+_l\d+_d\d+\.npz$")),
]

# Scalars worth lifting out of a cell into its manifest entry. Anything else
# stored in the file stays in the file; this is an index, not a copy.
PARAM_KEYS = ("kind", "lam0", "rho0", "ell", "d", "n", "seed", "n_steps", "lr",
              "flow_time", "log_every", "alpha", "rho_true", "q", "u0_fixed",
              "u_star", "lam_star", "phase", "sigma_eta_crit", "n_seeds",
              "tau", "frac_pos")


def family_of(name):
    """Which family a filename belongs to, or ``None`` if unrecognised."""
    for family, pattern in FAMILIES:
        if pattern.match(name):
            return family
    return None


def git_state(repo=None):
    """``(commit, dirty)`` for the working tree, or ``(None, None)`` if unknown."""
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout
        return sha, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _scalars(f):
    """The scalar metadata a cell stores about itself."""
    out = {}
    for k in PARAM_KEYS:
        if k in f.files:
            v = f[k]
            if v.ndim == 0:
                out[k] = v.item()
    return out


def dataset_gb(params):
    """Size of the ``(ell, n, d)`` training array in GB, the binding cost here.

    Returned rather than measured: the array is freed after each cell, so this
    is what the run *needed*, not what the file holds.
    """
    try:
        return round(params["ell"] * params["n"] * params["d"] * 8 / 1e9, 4)
    except KeyError:
        return None


def entry_for(path, family=None, duration_s=None, source="backfilled",
              commit=None, dirty=None):
    """Build a manifest entry by reading a cell's own stored metadata."""
    path = Path(path)
    with np.load(path) as f:
        params = _scalars(f)
        arrays = {k: list(f[k].shape) for k in f.files if f[k].ndim > 0}
    return {
        "file": path.name,
        "family": family or family_of(path.name),
        "params": params,
        "arrays": arrays,
        "dataset_gb": dataset_gb(params),
        "file_bytes": path.stat().st_size,
        "duration_s": duration_s,
        "commit": commit,
        "dirty": dirty,
        "source": source,
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def manifest_path(data_dir):
    return Path(data_dir) / MANIFEST_NAME


def load(data_dir):
    """Every entry recorded for one data directory, oldest first."""
    p = manifest_path(data_dir)
    if not p.exists():
        return []
    with p.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _append(data_dir, entries):
    p = manifest_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        for e in entries:
            fh.write(json.dumps(e, sort_keys=True) + "\n")


def record(path, duration_s, family=None, data_dir=None):
    """Record a cell a generator has just written.

    Called after the file exists, so the entry is read back from it rather
    than passed in -- the manifest then cannot drift from the data. Failing
    to record must never lose the cell itself, so errors here are swallowed
    with a warning.
    """
    path = Path(path)
    data_dir = Path(data_dir) if data_dir else path.parent
    commit, dirty = git_state()
    try:
        _append(data_dir, [entry_for(path, family=family, duration_s=duration_s,
                                     source="generated", commit=commit, dirty=dirty)])
    except Exception as exc:  # noqa: BLE001 - never lose a cell over bookkeeping
        print(f"  [manifest] could not record {path.name}: {exc}", file=sys.stderr)


def durations_from_logs(log_paths):
    """Map filename -> seconds, parsed from generator logs.

    The generators print ``wrote NAME (...) in  12.3s``. Reading that back is
    the only way to recover the cost of cells produced before this module
    existed, and those are the expensive ones.
    """
    pattern = re.compile(r"wrote\s+(\S+\.npz)\s+\([^)]*\)\s+in\s+([\d.]+)s")
    out = {}
    for p in log_paths:
        try:
            text = Path(p).read_text()
        except OSError:
            continue
        for name, secs in pattern.findall(text):
            out[name] = float(secs)   # a re-run overwrites; last wins
    return out


def backfill(data_dir, durations=None, verbose=True):
    """Add entries for files on disk that the manifest does not yet mention.

    Provenance is left null for these: the commit that produced them is not
    recoverable after the fact, and guessing it would be worse than admitting
    it is unknown.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    known = {e["file"] for e in load(data_dir)}
    durations = durations or {}

    new = []
    for p in sorted(data_dir.glob("*.npz")):
        if p.name in known:
            continue
        family = family_of(p.name)
        if family is None:
            if verbose:
                print(f"  skipping unrecognised file {p.name}")
            continue
        new.append(entry_for(p, family=family, duration_s=durations.get(p.name),
                             source="backfilled"))
    if new:
        _append(data_dir, new)
    if verbose:
        timed = sum(e["duration_s"] is not None for e in new)
        print(f"  {data_dir.name}: added {len(new)} entries "
              f"({timed} with a recovered duration), {len(known)} already present")
    return new


def summary(data_dirs=None, verbose=True):
    """Per-family totals: cell count, wall-clock cost, largest dataset."""
    data_dirs = data_dirs or DATA_DIRS
    rows = {}
    for d in data_dirs:
        for e in load(d):
            r = rows.setdefault(e["family"], {"cells": 0, "seconds": 0.0,
                                              "untimed": 0, "max_gb": 0.0,
                                              "bytes": 0})
            r["cells"] += 1
            r["bytes"] += e.get("file_bytes") or 0
            if e.get("duration_s") is None:
                r["untimed"] += 1
            else:
                r["seconds"] += e["duration_s"]
            r["max_gb"] = max(r["max_gb"], e.get("dataset_gb") or 0.0)

    if verbose:
        print(f"{'family':<20}{'cells':>7}{'untimed':>9}{'cost':>12}"
              f"{'peak data':>12}{'on disk':>10}")
        for family in sorted(rows):
            r = rows[family]
            cost = (f"{r['seconds'] / 3600:.2f} h" if r["seconds"] >= 3600
                    else f"{r['seconds']:.0f} s")
            print(f"{family:<20}{r['cells']:>7}{r['untimed']:>9}{cost:>12}"
                  f"{r['max_gb']:>10.1f} GB{r['bytes'] / 1e6:>7.1f} MB")
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "summary"
    logs = [argv[i + 1] for i, a in enumerate(argv) if a == "--log"]

    if cmd == "backfill":
        durations = durations_from_logs(logs) if logs else {}
        if logs:
            print(f"recovered {len(durations)} durations from {len(logs)} log(s)")
        for d in DATA_DIRS:
            backfill(d, durations=durations)
        print()
        summary()
    elif cmd == "summary":
        summary()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
