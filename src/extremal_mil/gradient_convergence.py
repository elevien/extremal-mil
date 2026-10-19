"""Data generation for the SI's initial-gradient convergence figure.

Measures how the finite-size :math:`\\dot\\lambda` approaches the asymptotic
gradient-flow value as :math:`d` grows at fixed :math:`\\alpha`, at a single
point in :math:`(\\lambda, u, \\rho)`. Nothing is integrated: the empirical
side is :func:`extremal_mil.model.loss_and_grad` evaluated at that point and
the asymptotic side is :func:`extremal_mil.ode.gradient_flow_rhs` at the same
point, so no learning rate enters.

Two things distinguish this from the sweeps in
:mod:`extremal_mil.si_experiment`.

*Replicates.* Each :math:`(d, u)` cell is repeated over several dataset
draws. The quantity plotted is an absolute difference, which has a positive
expectation even when the bias is zero, so the scatter has to be visible
alongside the mean.

*Streaming.* :math:`\\partial L/\\partial\\lambda` is a mean over bags, so it
accumulates bag by bag and the bags never need to exist at once. Peak memory
is then set by the chunk size rather than by :math:`\\ell`, which is what makes
:math:`d>225` reachable: at :math:`d=270`, :math:`n=729416` and the full
:math:`\\ell=100` dataset would be 158 GB. :func:`grad_lam` is checked against
the in-memory :func:`extremal_mil.convergence.empirical_gradient` in
``tests``-style form by :func:`check_streaming`.

``lambda0 = 0.1`` rather than the 0.2 used elsewhere in the SI. In HR the
asymptotic :math:`\\dot\\lambda \\propto \\sqrt{2\\alpha}\\rho_{\\rm true} -
\\lambda\\rho` vanishes at :math:`\\lambda_0 = 0.2232`, so at 0.2 the two
panels would be measured against quantities a factor of eight apart.

Run as a script to generate every cell:

    OMP_NUM_THREADS=1 python -m extremal_mil.gradient_convergence

# NEEDS HUMAN REVIEW
"""

import time
from pathlib import Path

import numpy as np

from . import convergence as cv
from . import manifest, model, ode

ALPHA = 0.05
RHO_TRUE = 1.2
Q = 0.2
LAM0 = 0.1
RHO0 = 1.7
ELL = 100

U_VALS = (0.1, 0.7)

# d up to 225 is affordable in memory, so it gets eight draws; above that each
# cell costs minutes and gets four.
D_SMALL = [22, 46, 77, 100, 150, 180, 195, 210, 225]
D_LARGE = [240, 255, 270]
K_SMALL = 8
K_LARGE = 4

# Largest array the streamed gradient will hold at once. The chunk is as many
# bags as fit in this; one bag is n*d*8 bytes, 3.5 GB at d=285, which is where
# this approach runs out on a 32 GB machine.
MAX_CHUNK_GB = 6.0

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "notebooks" / "si_data"
)


def n_from_d(d, alpha=ALPHA):
    return int(round(np.exp(alpha * d)))


def seed_for(d, rep):
    """Dataset seed for one replicate, matching the committed figure."""
    return 910000 + 1000 * rep + d


def theory_lam(u, lam0=LAM0, rho0=RHO0, alpha=ALPHA, rho_true=RHO_TRUE, q=Q):
    """Asymptotic :math:`\\dot\\lambda` at the evaluation point."""
    return float(ode.gradient_flow_rhs(0.0, [lam0, u, rho0], alpha, rho_true, q)[0])


def grad_lam(d, u, seed, ell=ELL, lam0=LAM0, rho0=RHO0,
             alpha=ALPHA, rho_true=RHO_TRUE, q=Q, max_chunk_gb=MAX_CHUNK_GB):
    """Empirical :math:`\\dot\\lambda` at ``(lam0, u, rho0)``, streaming the bags.

    Equivalent to taking the first component of
    :func:`extremal_mil.convergence.empirical_gradient` on the whole dataset,
    but the bags are drawn and consumed in chunks so the full ``(ell, n, d)``
    array is never allocated.
    """
    n = n_from_d(d, alpha)
    rng = np.random.default_rng(seed)
    b_true, e1, delta = cv.make_frame(d, rng)
    beta_true = rho_true * b_true
    b = cv.b_of_u(u, e1, delta, d)

    per_bag_gb = n * d * 8 / 1e9
    chunk = max(1, int(max_chunk_gb / per_bag_gb))

    total, done = 0.0, 0
    while done < ell:
        m = min(chunk, ell - done)
        X = rng.standard_normal((m, n, d))
        is_pos = rng.random(m) < q
        X[is_pos, 0, :] = beta_true
        w = (X @ beta_true).max(axis=1)

        z = X @ b
        sigma = model.softmax(lam0 * z, axis=-1)
        zbar = (sigma * z).sum(-1)
        err = rho0 * zbar - w
        var_sigma = (sigma * z ** 2).sum(-1) - zbar ** 2
        # grad_lam = sum_k 2 err_k rho Var_k / (ell d^2); accumulate the sum
        total += float(np.sum(2.0 * err * rho0 * var_sigma))
        done += m
        del X, z, sigma, w
    return -total / (ell * d ** 2)


def check_streaming(d=150, u=0.7, seed=12345, tol=1e-9):
    """Confirm :func:`grad_lam` matches the in-memory gradient at one ``d``.

    The streamed and in-memory routines draw from the random stream in a
    different order, so they are compared on a dataset built the same way
    rather than on the same seed.
    """
    from . import data

    n = n_from_d(d)
    rng = np.random.default_rng(seed)
    b_true, e1, delta = cv.make_frame(d, rng)
    X, w, _ = data.make_dataset(ELL, n, d, b_true, RHO_TRUE, Q, rng)
    ref = float(cv.empirical_gradient(X, w, LAM0, u, RHO0, e1, delta, d)[0])
    del X

    # same arithmetic, one chunk, on the same arrays
    rng = np.random.default_rng(seed)
    b_true, e1, delta = cv.make_frame(d, rng)
    X, w, _ = data.make_dataset(ELL, n, d, b_true, RHO_TRUE, Q, rng)
    b = cv.b_of_u(u, e1, delta, d)
    z = X @ b
    sigma = model.softmax(LAM0 * z, axis=-1)
    zbar = (sigma * z).sum(-1)
    err = RHO0 * zbar - w
    var_sigma = (sigma * z ** 2).sum(-1) - zbar ** 2
    streamed = -float(np.sum(2.0 * err * RHO0 * var_sigma)) / (ELL * d ** 2)
    del X

    assert abs(streamed - ref) < tol, (streamed, ref)
    return streamed, ref


def _path(data_dir, d):
    return Path(data_dir) / f"gradconv_l{ELL}_lam{LAM0:g}_d{d}.npz"


def _run_cell(data_dir, d, n_reps, verbose=True):
    """Compute and save one ``d`` cell, unless it is already on disk."""
    path = _path(data_dir, d)
    if path.exists():
        return
    t0 = time.time()
    n = n_from_d(d)
    err = np.empty((len(U_VALS), n_reps))
    for j, u in enumerate(U_VALS):
        th = theory_lam(u)
        for s in range(n_reps):
            err[j, s] = abs(grad_lam(d, u, seed_for(d, s)) - th)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        d=d, n=n, ell=ELL, lam0=LAM0, rho0=RHO0, alpha=ALPHA,
        rho_true=RHO_TRUE, q=Q, n_reps=n_reps,
        u=np.array(U_VALS), seeds=np.array([seed_for(d, s) for s in range(n_reps)]),
        theory=np.array([theory_lam(u) for u in U_VALS]),
        err=err,
    )
    elapsed = time.time() - t0
    manifest.record(path, duration_s=elapsed)
    if verbose:
        msg = "  ".join(f"u={u}: {err[j].mean():.4f}" for j, u in enumerate(U_VALS))
        print(f"  wrote {path.name} (n={n}) {msg}  in {elapsed:6.1f}s", flush=True)


def generate_all(data_dir=None, verbose=True):
    """Every cell, cheapest first so an interrupted run still leaves the
    small-``d`` end complete."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if verbose:
        print(f"lambda0={LAM0}  rho0={RHO0}  alpha={ALPHA}  "
              f"rho_true={RHO_TRUE}  q={Q}  ell={ELL}")
        print(f"theory: " + "  ".join(f"u={u}: {theory_lam(u):+.4f}" for u in U_VALS))
        print(f"writing to {data_dir}\n")
    for d in D_SMALL:
        _run_cell(data_dir, d, K_SMALL, verbose)
    for d in D_LARGE:
        _run_cell(data_dir, d, K_LARGE, verbose)


def load(d, data_dir=None):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    with np.load(_path(data_dir, d)) as f:
        return {k: f[k] for k in f.files}


def available(data_dir=None):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    out = []
    for p in sorted(Path(data_dir).glob("gradconv_l*_lam*_d*.npz")):
        with np.load(p) as f:
            out.append(int(f["d"]))
    return sorted(out)


if __name__ == "__main__":
    generate_all()
