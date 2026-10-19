"""Data generation for the Supplemental Material's numerical sections.

Three families, each one file per cell under ``notebooks/si_data/``:

``alpha``
    Gradient curves swept over the overlap ``u``, as in the main text's last
    figure, but repeated at several ``alpha``. Since ``n = e^{alpha d}``,
    changing alpha at fixed ``d`` changes the bag size, and it moves both
    phase boundaries: ``lambda* = sqrt(2 alpha)`` and
    ``u* = phi_alpha(lambda0)/rho_true``.

``lam0``
    The same sweep repeated at several initial temperatures ``lambda0``,
    spanning the H/L boundary.

``lamsweep``
    The transposed sweep: ``dlambda/dt`` against ``lambda0`` at fixed ``u``,
    one value of ``u`` held inside R and one inside N. Which phase a point
    sits in is not fixed along such a sweep unless ``u`` is chosen for it, so
    the two values are checked against their bands at import.

Run as a script to generate every cell the SI figures plot:

    python -m extremal_mil.si_experiment

"""

import time
from pathlib import Path

import numpy as np

from . import convergence as cv
from . import manifest, theory

# Held at the main text's values except where a family sweeps them.
ALPHA = 0.05
RHO_TRUE = 1.2
Q = 0.2
RHO0 = 1.7
LAM0 = 0.2

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "si_data"

U_VALS = np.linspace(-0.95, 0.95, 41)
# Bag count for every SI cell, matched to the main-text figures.
ELL = 100
D_VALUES = [20, 60, 100, 150]

# alpha family. 0.05 is the main text's value; the other two move lambda*
# (0.224, 0.316, 0.447) while keeping alpha < rho_true^2/2 = 0.72, so the
# planted instance stays identifiable throughout.
#
# Each alpha gets its own d grid, chosen so that n = e^{alpha d} takes the
# same four values 3, 20, 148, 1808 at every alpha. Holding n rather than d
# is what isolates alpha: a shared d grid would change the bag size along
# with the phase boundaries, and at alpha=0.10 it is not affordable anyway --
# d=100 gives n=22026 and a 49 GB dataset.
ALPHA_VALUES = [0.025, 0.05, 0.10]
D_VALUES_ALPHA = {
    0.025: [40, 120, 200, 300],
    0.05: [20, 60, 100, 150],
    0.10: [10, 30, 50, 75],
}

# lam0 family, spanning the H/L boundary at lambda* = sqrt(2 alpha) = 0.316.
LAM0_VALUES = [0.05, 0.2, 0.5]

# lamsweep family. Retrieval needs rho_true*u > phi_alpha(lambda), and
# phi_alpha falls to sqrt(2 alpha) at lambda* and is flat above, so over
# LAM_VALS a u stays in R throughout only if rho_true*u > phi_alpha(min) and
# in N throughout only if rho_true*u < sqrt(2 alpha). The two values below sit
# inside those bands and are asserted against them at import.
LAM_VALS = np.linspace(0.10, 0.40, 41)
U_RETRIEVAL = 0.7
U_NON_RETRIEVAL = 0.1
D_VALUES_LAM = [20, 60, 100, 150, 180, 195, 210, 225]



def _check_lam_sweep_u():
    """Fail loudly if the fixed-u values do not hold their phase over LAM_VALS."""
    phi = [theory.phi_alpha(l, ALPHA) for l in LAM_VALS]
    assert RHO_TRUE * U_RETRIEVAL > max(phi), (
        f"u={U_RETRIEVAL} leaves R: needs u > {max(phi) / RHO_TRUE:.3f}")
    assert RHO_TRUE * U_NON_RETRIEVAL < theory.lambda_star(ALPHA), (
        f"u={U_NON_RETRIEVAL} leaves N: needs u < "
        f"{theory.lambda_star(ALPHA) / RHO_TRUE:.3f}")


_check_lam_sweep_u()


def n_from_d(d, alpha=ALPHA):
    return int(round(np.exp(alpha * d)))


def u_star(lam0=LAM0, alpha=ALPHA, rho_true=RHO_TRUE):
    return theory.phi_alpha(lam0, alpha) / rho_true


def x_gb(ell, d, alpha=ALPHA):
    return ell * n_from_d(d, alpha) * d * 8 / 1e9


# ---------------------------------------------------------------------------
# alpha and lam0 families: gradient curves over u.
# ---------------------------------------------------------------------------
def _u_path(data_dir, alpha, lam0, d, ell=ELL):
    return Path(data_dir) / f"si_u_a{alpha:g}_lam{lam0:g}_l{ell}_d{d}.npz"


def _run_u_cell(data_dir, alpha, lam0, d, seed, verbose, ell=ELL):
    path = _u_path(data_dir, alpha, lam0, d, ell)
    if path.exists():
        return
    n = n_from_d(d, alpha)
    t0 = time.time()
    emp, th = cv.gradient_curves_vs_u(
        U_VALS, lam0, RHO0, ell, n, d, np.random.default_rng(seed),
        alpha=alpha, rho_true=RHO_TRUE, q=Q)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, u=U_VALS, alpha=alpha, lam0=lam0, rho0=RHO0, ell=ell, d=d,
             n=n, seed=seed, rho_true=RHO_TRUE, q=Q,
             lam_star=theory.lambda_star(alpha),
             u_star=u_star(lam0, alpha),
             **{f"emp_{k}": v for k, v in emp.items()},
             **{f"th_{k}": v for k, v in th.items()})
    el = time.time() - t0
    manifest.record(path, duration_s=el)
    if verbose:
        print(f"  wrote {path.name} (n={n}) in {el:6.1f}s", flush=True)


# ---------------------------------------------------------------------------
# lamsweep family: dlambda/dt over lambda0 at fixed u.
# ---------------------------------------------------------------------------
def _lam_path(data_dir, u0, d, ell=ELL):
    return Path(data_dir) / f"si_lam_u{u0:g}_l{ell}_d{d}.npz"


def _run_lam_cell(data_dir, u0, d, seed, verbose, ell=ELL):
    path = _lam_path(data_dir, u0, d, ell)
    if path.exists():
        # The filename carries u and d but not the lambda grid, so a cell left
        # over from a different LAM_VALS would be reused silently. Compare.
        with np.load(path) as f:
            if f["lam"].shape == LAM_VALS.shape and np.allclose(f["lam"], LAM_VALS):
                return
    n = n_from_d(d)
    t0 = time.time()
    emp, th = cv.gradient_curves_vs_lam(
        LAM_VALS, u0, RHO0, ell, n, d, np.random.default_rng(seed),
        ALPHA, RHO_TRUE, Q)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, lam=LAM_VALS, u0=u0, rho0=RHO0, ell=ell, d=d, n=n,
             seed=seed, alpha=ALPHA, rho_true=RHO_TRUE, q=Q,
             lam_star=theory.lambda_star(ALPHA),
             phase=("R" if RHO_TRUE * u0 > theory.lambda_star(ALPHA) else "N"),
             **{f"emp_{k}": v for k, v in emp.items()},
             **{f"th_{k}": v for k, v in th.items()})
    el = time.time() - t0
    manifest.record(path, duration_s=el)
    if verbose:
        print(f"  wrote {path.name} (n={n}) in {el:6.1f}s", flush=True)


def generate_all(data_dir=None, verbose=True):
    """Run every cell the SI figures plot: the largest d of each alpha and
    lam0 sweep, and the full d grid of the lambda0 sweep, at ``ELL``."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if verbose:
        print(f"writing to {data_dir}")
        print(f"  alpha  {ALPHA_VALUES}  (lambda* = "
              f"{[round(float(theory.lambda_star(a)), 3) for a in ALPHA_VALUES]})")
        print(f"  lam0   {LAM0_VALUES}   (lambda* = "
              f"{theory.lambda_star(ALPHA):.3f})")
        print(f"  d      {D_VALUES_LAM} for the lambda0 sweep, ell={ELL}\n")
        print("alpha sweep:")
    # The alpha sweep runs first: its alpha = 0.05 cell is also the lam0 = 0.2
    # cell of the lam0 sweep, and is kept with the alpha-sweep seed.
    for a in ALPHA_VALUES:
        d = D_VALUES_ALPHA[a][-1]
        _run_u_cell(data_dir, a, LAM0, d, seed=500000 + int(1e4 * a) + d,
                    verbose=verbose)
    if verbose:
        print("\nlambda0 sweep:")
    for lam0 in LAM0_VALUES:
        d = D_VALUES[-1]
        _run_u_cell(data_dir, ALPHA, lam0, d,
                    seed=600000 + int(1e4 * lam0) + d, verbose=verbose)
    if verbose:
        print("\ndlambda/dt against lambda0:")
    for u0 in (U_NON_RETRIEVAL, U_RETRIEVAL):
        for d in D_VALUES_LAM:
            _run_lam_cell(data_dir, u0, d,
                          seed=800000 + int(100 * u0) + d, verbose=verbose)


def load_u_curves(alpha, lam0, d, data_dir=None, ell=ELL):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    with np.load(_u_path(data_dir, alpha, lam0, d, ell)) as f:
        return {k: f[k] for k in f.files}


def load_lam_curves(u0, d, data_dir=None, ell=ELL):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    with np.load(_lam_path(data_dir, u0, d, ell)) as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    generate_all()
