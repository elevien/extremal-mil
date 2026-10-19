"""Data generation for Figures 2 and 3 (``notebooks/fig2.ipynb``, ``fig3.ipynb``).

For a grid of initial conditions, runs GD and logs the full
``(lambda, u, rho, loss)`` trajectory (not just the endpoint), for a grid of
``ell`` (bag count), ``d`` (dimension), and learning rate. Each
``(kind, ell, d, lr, n_steps)`` cell is saved to its own file under
``notebooks/fig2_data/``, so a cell already on disk is never recomputed.

Run as a script to generate every cell:

    python -m extremal_mil.fig2_experiment

"""

import time
from pathlib import Path

import numpy as np

from . import manifest, sweeps

# ---------------------------------------------------------------------------
# Problem parameters.
# ---------------------------------------------------------------------------
ALPHA = 0.05        # alpha = ln(n) / d, fixes bag size n from d
RHO_TRUE = 1.2       # ||beta_true|| / ||b_true||
Q = 0.2              # probability a bag contains the planted/marked instance
LAM0 = 0.2           # initial inverse temperature (softmax sharpness)
RHO0 = 1.7           # initial output scale

# Each cell's file lives here, alongside the notebook that plots it.
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "notebooks" / "fig2_data"
)

# Sweeps are memory-bandwidth bound at large (ell, d); more workers stop
# helping well before the core count. Tuned on a 12-core machine.
N_WORKERS = 6

# Initial-condition grids.
U0_VALS = np.linspace(-0.9, 0.9, 11)
RHO0_VALS = np.linspace(0.2, 3.2, 11)
U0_FIXED = 0.54      # fixed u0 for the rho sweep, past the retrieval threshold

# The cells plotted in Figs. 2 and 3: two bag sizes at ell = 100, run to
# flow time t = lr * n_steps = 15000.
ELL_VALUES_TRAJ = [100]
D_VALUES_TRAJ = [22, 150]
LR_STEPS_TRAJ = [(0.1, 150000)]
LOG_EVERY_TRAJ = 200


def _u_inits():
    """(lam0, u0, rho0) triples for the u-sweep, at fixed rho0."""
    return [(LAM0, u0, RHO0) for u0 in U0_VALS]


def _rho_inits():
    """(lam0, u0, rho0) triples for the rho-sweep, at fixed u0=U0_FIXED."""
    return [(LAM0, U0_FIXED, r0) for r0 in RHO0_VALS]


def _fmt_lr(lr):
    """Learning rate as a filename-safe string, e.g. 0.1 -> '0.1'."""
    return f"{lr:g}"


def _traj_path(data_dir, kind, ell, d, lr, n_steps):
    """File for one (kind, ell, d, lr, n_steps) trajectory cell."""
    return (Path(data_dir) /
            f"traj_{kind}_l{ell}_d{d}_lr{_fmt_lr(lr)}_s{n_steps}.npz")


def workers_for(ell, d, n_workers=None):
    """Worker count for one (ell, d) cell, backed off once the shared
    dataset is large so each worker still has room for its own arrays."""
    if n_workers is None:
        n_workers = N_WORKERS
    x_gb = ell * sweeps.n_from_d(d, ALPHA) * d * 8 / 1e9
    return 4 if x_gb > 8.0 else n_workers


def _run_cell(data_dir, kind, ell, d, lr, n_steps, seed, n_workers, verbose):
    """Compute and save one trajectory cell, unless it is already on disk."""
    path = _traj_path(data_dir, kind, ell, d, lr, n_steps)
    if path.exists():
        return

    inits = _u_inits() if kind == "u" else _rho_inits()
    x0 = U0_VALS if kind == "u" else RHO0_VALS
    t0 = time.time()
    res, n = sweeps.trajectory_dataset(
        ell, d, ALPHA, RHO_TRUE, Q, inits, n_steps, lr, LOG_EVERY_TRAJ,
        data_seed=seed, n_workers=workers_for(ell, d, n_workers),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        kind=kind, x0_name=("u0" if kind == "u" else "rho0"), x0=x0,
        ell=ell, d=d, n=n, seed=seed, n_steps=n_steps, lr=lr,
        log_every=LOG_EVERY_TRAJ, flow_time=lr * n_steps,
        alpha=ALPHA, rho_true=RHO_TRUE, q=Q, lam0=LAM0, rho0=RHO0, u0_fixed=U0_FIXED,
        step=res["step"], t=lr * res["step"],
        lam=res["lam"], u=res["u"], rho=res["rho"], loss=res["loss"],
    )
    elapsed = time.time() - t0
    # Recorded in notebooks/fig2_data/manifest.jsonl
    manifest.record(path, duration_s=elapsed)
    if verbose:
        print(f"  wrote {path.name} (n={n}) in {elapsed:6.1f}s")


def generate_all(data_dir=None, n_workers=None, verbose=True):
    """Run every trajectory cell for Figures 2 and 3: the u0 sweep at each
    (ell, d, lr, n_steps) above.

    Safe to re-run or interrupt -- each cell is skipped if its file already
    exists.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if verbose:
        print(f"writing trajectory data to {data_dir}")
        print(f"  ell   {ELL_VALUES_TRAJ}")
        print(f"  d     {D_VALUES_TRAJ}")
        print(f"  lr    {[f'{lr} x {ns} steps (t={lr * ns:g})' for lr, ns in LR_STEPS_TRAJ]}")

    for lr, n_steps in LR_STEPS_TRAJ:
        for kind in ("u",):
            for ell in ELL_VALUES_TRAJ:
                if verbose:
                    print(f"\nkind={kind}  ell={ell}  lr={lr}  n_steps={n_steps}")
                for d in D_VALUES_TRAJ:
                    # Dataset seed depends on d only, matching the cells in
                    # the paper.
                    seed = 200000 + d
                    _run_cell(data_dir, kind, ell, d, lr, n_steps, seed,
                              n_workers, verbose)


def load_trajectory_sweep(kind, ell, d, lr, n_steps, data_dir=None):
    """Load one trajectory cell.

    Returns a dict with ``x0`` (the initial-condition grid), ``step`` and
    ``t = lr * step``, and ``lam``, ``u``, ``rho``, ``loss`` as
    ``(len(x0), n_points)`` arrays
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    with np.load(_traj_path(data_dir, kind, ell, d, lr, n_steps)) as f:
        return {k: f[k] for k in f.files}


def list_trajectory_sweeps(data_dir=None):
    """List the (kind, ell, d, lr, n_steps) trajectory cells on disk."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    out = []
    for p in sorted(data_dir.glob("traj_*_l*_d*_lr*_s*.npz")):
        with np.load(p) as f:
            out.append((str(f["kind"]), int(f["ell"]), int(f["d"]),
                        float(f["lr"]), int(f["n_steps"])))
    return out


if __name__ == "__main__":
    generate_all()
