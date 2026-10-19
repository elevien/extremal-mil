"""Helpers comparing the finite-(n,d,l) training gradient of ``extremal_mil``
against the asymptotic gradient-flow field ``ode.gradient_flow_rhs``.

Used by the Figure 4 and SI data generators.
"""
import numpy as np

from . import data, model, ode, theory

ALPHA = 0.05
RHO_TRUE = 1.2
Q = 0.5

# One representative (lambda, u, rho) evaluation point in each of the four
# phases HR / LR / HN / LN, used by the aggregate error sweeps below.
EVAL_POINTS = {
    "HR": (0.20, 0.50, 0.8),
    "LR": (0.60, 0.50, 0.8),
    "HN": (0.20, 0.10, 0.8),
    "LN": (0.60, 0.10, 0.8),
}


def b_of_u(u, e1, delta, d):
    u = np.clip(u, -1.0, 1.0)
    return np.sqrt(d) * (u * e1 + np.sqrt(1.0 - u ** 2) * delta)


def db_du(u, e1, delta, d):
    u = np.clip(u, -0.999, 0.999)
    return np.sqrt(d) * (e1 - (u / np.sqrt(1.0 - u ** 2)) * delta)


def make_frame(d, rng):
    """Return (b_true, e1, delta) for a bag of dimension d: a fixed true
    direction and a fixed direction delta perpendicular to it, spanning the
    plane the overlap sweep u -> b_of_u(u) moves in."""
    b_true = data.make_b_true(d)
    e1 = b_true / np.linalg.norm(b_true)
    delta = rng.standard_normal(d)
    delta -= (delta @ e1) * e1
    delta /= np.linalg.norm(delta)
    return b_true, e1, delta


def empirical_gradient(X, w, lam, u, rho, e1, delta, d, loss_and_grad=None):
    """(dlam/dt, du/dt, drho/dt) from the finite-(l,n,d) dataset (X, w).

    ``loss_and_grad`` defaults to the squared-error loss; pass
    :func:`extremal_mil.logistic.make_loss_and_grad` with ``w`` holding binary
    labels for the classification variant.
    """
    loss_and_grad = loss_and_grad or model.loss_and_grad
    b = b_of_u(u, e1, delta, d)
    _, grad_b, grad_lam, grad_rho = loss_and_grad(X, w, b, lam, rho)
    dlam = -grad_lam
    du = (1.0 - u ** 2) * (-(grad_b @ db_du(u, e1, delta, d)))
    drho = -grad_rho
    return np.array([dlam, du, drho])


def theory_gradient(lam, u, rho, alpha=ALPHA, rho_true=RHO_TRUE, q=Q):
    """(dlam/dt, du/dt, drho/dt) from the asymptotic gradient-flow field."""
    return np.array(ode.gradient_flow_rhs(0.0, [lam, u, rho], alpha, rho_true, q))


def relative_error(l, n, d, rng, eval_points=None, alpha=ALPHA, rho_true=RHO_TRUE, q=Q):
    """RMS relative error between empirical and theory gradients, averaged
    over ``eval_points`` (default: one point per phase), for one dataset
    draw of size (l, n, d)."""
    eval_points = eval_points or EVAL_POINTS
    b_true, e1, delta = make_frame(d, rng)
    X, w, _ = data.make_dataset(l, n, d, b_true, rho_true, q, rng)

    errs = []
    for lam, u, rho in eval_points.values():
        g_emp = empirical_gradient(X, w, lam, u, rho, e1, delta, d)
        g_th = theory_gradient(lam, u, rho, alpha, rho_true, q)
        errs.append(np.linalg.norm(g_emp - g_th) / np.linalg.norm(g_th))
    return float(np.mean(errs))


def error_sweep(values, make_l_n_d, n_seeds, seed0=0, **kwargs):
    """relative_error(...) averaged over n_seeds random datasets, for each
    value in ``values`` (with (l, n, d) given by make_l_n_d(value))."""
    means, sems = [], []
    for v in values:
        l, n, d = make_l_n_d(v)
        errs = [relative_error(l, n, d, np.random.default_rng(seed0 + s), **kwargs)
                 for s in range(n_seeds)]
        means.append(np.mean(errs))
        sems.append(np.std(errs) / np.sqrt(n_seeds))
    return np.array(means), np.array(sems)


def gradient_curves_vs_u(u_vals, lam0, rho0, l, n, d, rng,
                          alpha=ALPHA, rho_true=RHO_TRUE, q=Q,
                          loss_and_grad=None, theory_rhs=None, labels="w"):
    """Empirical and theory ``(dlam/dt, du/dt, drho/dt)`` over a grid of ``u``
    at fixed ``(lambda0, rho0)``, for one dataset draw of size ``(l, n, d)``.

    The direct, curve-level counterpart of :func:`error_sweep`'s aggregate
    scalar error metric. All three components are returned; the theory field
    does not depend on ``(l, n, d)``, so its curves are identical across
    draws and are returned only for convenience.

    ``loss_and_grad``, ``theory_rhs`` and ``labels`` select the problem:
    the defaults give squared-error regression against ``w``, while
    ``labels="z"`` with the logistic loss and
    :func:`extremal_mil.logistic.gradient_flow_rhs` give classification
    against the binary containment indicator.

    Returns
    -------
    (dict, dict)
        ``(empirical, theory)``, each with keys ``lam``, ``u``, ``rho``
        holding arrays shaped like ``u_vals``.
    """
    b_true, e1, delta = make_frame(d, rng)
    X, w, is_pos = data.make_dataset(l, n, d, b_true, rho_true, q, rng)
    target = is_pos.astype(float) if labels == "z" else w

    keys = ("lam", "u", "rho")
    emp = {k: np.empty_like(u_vals) for k in keys}
    th = {k: np.empty_like(u_vals) for k in keys}
    for i, u in enumerate(u_vals):
        g_emp = empirical_gradient(X, target, lam0, u, rho0, e1, delta, d,
                                   loss_and_grad=loss_and_grad)
        g_th = (np.array(theory_rhs(0.0, [lam0, u, rho0])) if theory_rhs
                else theory_gradient(lam0, u, rho0, alpha, rho_true, q))
        for j, k in enumerate(keys):
            emp[k][i], th[k][i] = g_emp[j], g_th[j]
    return emp, th


def gradient_curves_vs_lam(lam_vals, u, rho0, l, n, d, rng,
                           alpha=ALPHA, rho_true=RHO_TRUE, q=Q):
    """Empirical and theory ``(dlam/dt, du/dt, drho/dt)`` over a grid of
    ``lambda`` at fixed ``(u, rho0)``, for one dataset draw of size ``(l, n, d)``.

    The transpose of :func:`gradient_curves_vs_u`: there the temperature is
    held and the overlap swept, here the overlap is held and the temperature
    swept. Note that which phase the point sits in is not fixed along this
    sweep unless ``u`` is chosen for it -- retrieval needs
    ``rho_true * u > phi_alpha(lambda)``, and ``phi_alpha`` varies with
    ``lambda``, reaching its minimum ``sqrt(2 alpha)`` at ``lambda*``. A ``u``
    stays in R across the whole grid only if
    ``rho_true * u > max_lambda phi_alpha(lambda)``, and in N only if
    ``rho_true * u < sqrt(2 alpha)``.

    Returns
    -------
    (dict, dict)
        ``(empirical, theory)``, each keyed ``lam``, ``u``, ``rho``.
    """
    b_true, e1, delta = make_frame(d, rng)
    X, w, _ = data.make_dataset(l, n, d, b_true, rho_true, q, rng)

    keys = ("lam", "u", "rho")
    emp = {k: np.empty_like(lam_vals) for k in keys}
    th = {k: np.empty_like(lam_vals) for k in keys}
    for i, lam in enumerate(lam_vals):
        g_emp = empirical_gradient(X, w, lam, u, rho0, e1, delta, d)
        g_th = theory_gradient(lam, u, rho0, alpha, rho_true, q)
        for j, k in enumerate(keys):
            emp[k][i], th[k][i] = g_emp[j], g_th[j]
    return emp, th
