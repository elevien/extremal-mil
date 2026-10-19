"""Parallel initial-condition sweeps of GD training, for Figure 2.

Figure 2 (``notebooks/fig2_ell_convergence.ipynb``) needs the *final*
:math:`(u,\\rho)` reached by :func:`extremal_mil.gd.train` from many initial
conditions, on one shared dataset, repeated for several :math:`(d,\\ell)`.
Each individual run is single-threaded (the per-step work is a
``(l, n, d)`` einsum, which OpenBLAS does not parallelize), so the sweep is
run across processes instead.

The dataset is published to a module-level global *before* the pool is
created and the pool uses the ``fork`` start method, so the ``(l, n, d)``
array is shared copy-on-write with the workers rather than pickled to each
of them -- which matters at :math:`\\ell=15{,}000`, where ``X`` alone is
several GB.
"""

import multiprocessing as mp
import os

import numpy as np

from . import data, gd, ode

# Dataset shared with fork()ed workers; see module docstring.
_SHARED = {}


def n_from_d(d, alpha):
    """Instances per bag at fixed :math:`\\alpha=\\ln(n)/d`, i.e. ``n = e^{alpha d}``."""
    return int(round(np.exp(alpha * d)))


def theory_finals(inits, alpha, rho_true, q, t_max=800.0, max_step=2.0):
    """Final ``(lambda, u, rho)`` of the gradient flow from each initial condition.

    Parameters
    ----------
    inits : sequence of (lam0, u0, rho0)
    alpha, rho_true, q : float
        Problem parameters, passed to :func:`extremal_mil.ode.integrate`.
    t_max : float
        Integrate over ``[0, t_max]``; long enough that the flow has settled.

    Returns
    -------
    dict of (m,) arrays
        Keys ``lam``, ``u``, ``rho``.
    """
    out = {k: np.empty(len(inits)) for k in ("lam", "u", "rho")}
    for i, ic in enumerate(inits):
        sol = ode.integrate(list(ic), [0, t_max], alpha, rho_true, q, max_step=max_step)
        out["lam"][i], out["u"][i], out["rho"][i] = sol.y[:, -1]
    return out


def _run_one(args):
    """Train from one initial condition on the forked-in dataset."""
    lam0, u0, rho0, n_steps, lr, b0_seed = args
    hist = gd.train(
        _SHARED["X"], _SHARED["w"], _SHARED["b_true"],
        u0=u0, lam0=lam0, rho0=rho0,
        n_steps=n_steps, lr_b=lr, lr_lam=lr, lr_rho=lr,
        log_every=n_steps, rng=np.random.default_rng(b0_seed),
        loss_and_grad=_SHARED.get("loss_and_grad"),
    )
    return hist["lam"][-1], hist["u"][-1], hist["rho"][-1]


def gd_finals(X, w, b_true, inits, n_steps, lr, b0_seed=1, n_workers=None):
    """Final ``(lambda, u, rho)`` of GD from each initial condition, in parallel.

    Parameters
    ----------
    X, w, b_true : arrays
        As returned by :func:`extremal_mil.data.make_dataset` /
        :func:`extremal_mil.data.make_b_true`.
    inits : sequence of (lam0, u0, rho0)
    n_steps : int
    lr : float
        Shared learning rate for ``b``, ``lambda``, ``rho``.
    b0_seed : int
        Seed for the random perpendicular direction of ``b0``.
    n_workers : int, optional
        Defaults to ``min(len(inits), os.cpu_count() - 2)``.

    Returns
    -------
    dict of (m,) arrays
        Keys ``lam``, ``u``, ``rho``.
    """
    tasks = [(lam0, u0, rho0, n_steps, lr, b0_seed) for lam0, u0, rho0 in inits]
    if n_workers is None:
        n_workers = max(1, min(len(tasks), (os.cpu_count() or 2) - 2))

    _SHARED.update(X=X, w=w, b_true=b_true)
    try:
        if n_workers == 1:
            results = [_run_one(t) for t in tasks]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(n_workers) as pool:
                results = pool.map(_run_one, tasks, chunksize=1)
    finally:
        _SHARED.clear()

    lam, u, rho = (np.array(v) for v in zip(*results))
    return {"lam": lam, "u": u, "rho": rho}


def sweep_dataset(l, d, alpha, rho_true, q, inits, n_steps, lr,
                  data_seed, b0_seed=1, n_workers=None):
    """Generate one ``(l, d)`` dataset and run :func:`gd_finals` on it.

    ``n`` is set from ``d`` by :func:`n_from_d`. The dataset is freed before
    returning, so successive ``(l, d)`` cells do not hold two copies of ``X``
    at once.

    Returns
    -------
    (dict, int)
        The :func:`gd_finals` result, and the ``n`` that was used.
    """
    n = n_from_d(d, alpha)
    b_true = data.make_b_true(d)
    X, w, _ = data.make_dataset(l, n, d, b_true, rho_true, q,
                                np.random.default_rng(data_seed))
    try:
        return gd_finals(X, w, b_true, inits, n_steps, lr,
                         b0_seed=b0_seed, n_workers=n_workers), n
    finally:
        del X, w


# ---------------------------------------------------------------------------
# Full trajectories (not just final points).
#
# gd_finals above keeps only the last (lam, u, rho) of each run. The functions
# below keep the whole logged history, parallelized the same way: one worker
# per initial condition, dataset shared copy-on-write through fork.
# ---------------------------------------------------------------------------
def _run_one_traj(args):
    """Train from one initial condition and return the full logged history."""
    lam0, u0, rho0, n_steps, lr, log_every, b0_seed = args
    hist = gd.train(
        _SHARED["X"], _SHARED["w"], _SHARED["b_true"],
        u0=u0, lam0=lam0, rho0=rho0,
        n_steps=n_steps, lr_b=lr, lr_lam=lr, lr_rho=lr,
        log_every=log_every, rng=np.random.default_rng(b0_seed),
        loss_and_grad=_SHARED.get("loss_and_grad"),
    )
    return hist


def gd_trajectories(X, w, b_true, inits, n_steps, lr, log_every,
                    b0_seed=1, n_workers=None, loss_and_grad=None):
    """Full ``(lambda, u, rho, loss)`` histories from each initial condition.

    ``loss_and_grad`` is forwarded to :func:`extremal_mil.gd.train`, so the
    classification variant runs through this sweep unchanged with ``w``
    holding binary labels.

    Returns
    -------
    dict
        ``step`` is ``(n_points,)``; ``lam``, ``u``, ``rho``, ``loss`` are
        each ``(len(inits), n_points)``.
    """
    tasks = [(lam0, u0, rho0, n_steps, lr, log_every, b0_seed)
             for lam0, u0, rho0 in inits]
    if n_workers is None:
        n_workers = max(1, min(len(tasks), (os.cpu_count() or 2) - 2))

    _SHARED.update(X=X, w=w, b_true=b_true, loss_and_grad=loss_and_grad)
    try:
        if n_workers == 1:
            hists = [_run_one_traj(t) for t in tasks]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(n_workers) as pool:
                hists = pool.map(_run_one_traj, tasks, chunksize=1)
    finally:
        _SHARED.clear()

    return {
        "step": hists[0]["step"],
        **{k: np.vstack([h[k] for h in hists]) for k in ("lam", "u", "rho", "loss")},
    }


def trajectory_dataset(l, d, alpha, rho_true, q, inits, n_steps, lr, log_every,
                       data_seed, b0_seed=1, n_workers=None):
    """Generate one ``(l, d)`` dataset and run :func:`gd_trajectories` on it.

    The trajectory counterpart of :func:`sweep_dataset`, including freeing the
    dataset before returning.

    Returns
    -------
    (dict, int)
        The :func:`gd_trajectories` result, and the ``n`` that was used.
    """
    n = n_from_d(d, alpha)
    b_true = data.make_b_true(d)
    X, w, _ = data.make_dataset(l, n, d, b_true, rho_true, q,
                                np.random.default_rng(data_seed))
    try:
        return gd_trajectories(X, w, b_true, inits, n_steps, lr, log_every,
                               b0_seed=b0_seed, n_workers=n_workers), n
    finally:
        del X, w
