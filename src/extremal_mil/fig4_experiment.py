"""Data generation for Figure 4 (``notebooks/fig4.ipynb``).

Figure 4 asks whether the finite-:math:`(n,d,\\ell)` training gradient
converges to the asymptotic gradient-flow field. Each cell sweeps the
overlap :math:`u` at a fixed :math:`(\\lambda_0,\\rho_0)` on one dataset draw
and records all three components of the empirical flow alongside the
:math:`d\\to\\infty` prediction, via
:func:`extremal_mil.convergence.gradient_curves_vs_u`.

The paper's figure sweeps ``d`` at fixed ``ell`` from the same
:math:`(\\lambda_0,\\rho_0)` as the trajectory figure. ``lam0 = 0.2`` is
below :math:`\\lambda^\\star=\\sqrt{2\\alpha}`, so the softmax has not
collapsed and :math:`\\dot\\lambda>0` across the grid.

Storage
-------
One self-describing file per ``(lam0, ell, d)`` cell,
``notebooks/fig4_data/curves_lam{lam0}_l{ell}_d{d}.npz``, holding the ``u``
grid, the empirical and theory components, and the parameters that produced
it. Loading is a plain ``np.load`` wrapped as :func:`load_curves`; a cell is
skipped if its file already exists, so an interrupted run resumes.

Usage
-----
Run

    python -m extremal_mil.fig4_experiment

"""

import time
from pathlib import Path

import numpy as np

from . import convergence as cv
from . import manifest, theory

# Matched to fig2_experiment, so Figs. 2 and 3 describe one system.
ALPHA = 0.05
RHO_TRUE = 1.2
Q = 0.2
RHO0 = 1.7

LAM_STAR = theory.lambda_star(ALPHA)

# Below lambda* (softmax not collapsed, lambda still moving) and above it
# (lambda frozen). The trajectory figure starts at the former.
LAM0_HIGH_T = 0.2
LAM0_LOW_T = 0.5

U_VALS = np.linspace(-0.95, 0.95, 41)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "fig4_data"

# The d sweep at fixed ell plotted in Figure 4.
ELL_MAIN = 100
D_VALUES = [20, 60, 100, 150]


def x_gb(ell, d, alpha=ALPHA):
    """Size of the ``(ell, n, d)`` dataset in GB, the binding constraint here."""
    return ell * int(round(np.exp(alpha * d))) * d * 8 / 1e9


def cells():
    """Every ``(lam0, ell, d)`` cell Figure 4 needs."""
    return [(LAM0_HIGH_T, ELL_MAIN, d) for d in D_VALUES]


def _path(data_dir, lam0, ell, d):
    return Path(data_dir) / f"curves_lam{lam0:g}_l{ell}_d{d}.npz"


def u_star(lam0, alpha=ALPHA, rho_true=RHO_TRUE):
    """Retrieval threshold in ``u`` at fixed ``lam0``."""
    return theory.phi_alpha(lam0, alpha) / rho_true


def _run_cell(data_dir, lam0, ell, d, seed, verbose):
    """Compute and save one ``(lam0, ell, d)`` cell, unless already on disk."""
    path = _path(data_dir, lam0, ell, d)
    if path.exists():
        return

    n = int(round(np.exp(ALPHA * d)))
    t0 = time.time()
    emp, th = cv.gradient_curves_vs_u(
        U_VALS, lam0, RHO0, ell, n, d, np.random.default_rng(seed),
        ALPHA, RHO_TRUE, Q)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        u=U_VALS, lam0=lam0, rho0=RHO0, ell=ell, d=d, n=n, seed=seed,
        alpha=ALPHA, rho_true=RHO_TRUE, q=Q,
        u_star=u_star(lam0), lam_star=LAM_STAR,
        emp_lam=emp["lam"], emp_u=emp["u"], emp_rho=emp["rho"],
        th_lam=th["lam"], th_u=th["u"], th_rho=th["rho"],
    )
    elapsed = time.time() - t0
    manifest.record(path, duration_s=elapsed)
    if verbose:
        print(f"  wrote {path.name} (n={n}, X={x_gb(ell, d):.1f} GB) "
              f"in {elapsed:6.1f}s", flush=True)


def generate_all(data_dir=None, verbose=True):
    """Run every cell of the grid, skipping those already on disk."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    todo = cells()
    if verbose:
        print(f"alpha={ALPHA}  rho_true={RHO_TRUE}  q={Q}  rho0={RHO0}")
        print(f"lambda* = {LAM_STAR:.4f}  (lam0={LAM0_HIGH_T} is high-T, "
              f"{LAM0_LOW_T} is low-T)")
        print(f"{len(todo)} cells -> {data_dir}\n")
    t0 = time.time()
    for lam0, ell, d in todo:
        # Seed depends on the cell, so every curve is an independent draw and
        # two cells that share (ell, d) across lam0 also share their dataset.
        _run_cell(data_dir, lam0, ell, d, seed=10 * ell + d,
                  verbose=verbose)
    if verbose:
        print(f"\ntotal {(time.time() - t0) / 60:.1f} min")


def load_curves(lam0, ell, d, data_dir=None):
    """Load one cell: ``{'u', 'emp_lam', 'th_lam', ..., + its metadata}``."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    with np.load(_path(data_dir, lam0, ell, d)) as f:
        return {k: f[k] for k in f.files}


def list_available(data_dir=None):
    """List the ``(lam0, ell, d)`` cells on disk."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    out = []
    for p in sorted(Path(data_dir).glob("curves_lam*_l*_d*.npz")):
        with np.load(p) as f:
            out.append((float(f["lam0"]), int(f["ell"]), int(f["d"])))
    return sorted(out)


if __name__ == "__main__":
    generate_all()
