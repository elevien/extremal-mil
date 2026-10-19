"""Projected gradient descent on (b, lambda, rho).

This is the discrete-time counterpart of the continuous-time flow
eq:gradient_flow_def in ``mil.tex``:

.. math::

    \\dot\\lambda = -\\partial_\\lambda L, \\qquad
    \\dot\\rho = -\\partial_\\rho L, \\qquad
    \\dot{\\bb} = -P_{\\bb}\\nabla_{\\bb} L,

where :math:`P_{\\bb} v = v - \\langle v,\\bb\\rangle\\bb/d` projects onto the
tangent space of the sphere :math:`\\|\\bb\\|^2=d`.
"""

import numpy as np

from .model import loss_and_grad as mse_loss_and_grad


def overlap(b, b_true):
    """Overlap :math:`u = \\langle\\bb,\\bb_{\\rm true}\\rangle/d` (eq:overlap_param)."""
    d = b.shape[0]
    return float(b @ b_true) / d


def make_b0(u0, b_true, rng):
    """Construct ``b0`` with ``||b0||^2 == d`` and overlap ``u0`` with ``b_true``.

    Follows eq:overlap_param: :math:`\\bb = \\sqrt{d}(u\\,e_1 + \\sqrt{1-u^2}\\,
    \\boldsymbol\\delta)` with :math:`\\boldsymbol\\delta \\perp e_1`,
    :math:`\\|\\boldsymbol\\delta\\|=1`.
    """
    d = b_true.shape[0]
    e1 = b_true / np.linalg.norm(b_true)
    delta = rng.standard_normal(d)
    delta -= (delta @ e1) * e1
    delta /= np.linalg.norm(delta)
    u0 = np.clip(u0, -1.0, 1.0)
    return np.sqrt(d) * (u0 * e1 + np.sqrt(1.0 - u0 ** 2) * delta)


def project_tangent(v, b):
    """Project ``v`` onto the tangent space of ``||b||^2 = d`` at ``b``."""
    d = b.shape[0]
    return v - (v @ b) / d * b


def train(X, w, b_true, u0, lam0, rho0,
          n_steps=5000, lr_b=1e-2, lr_lam=1e-2, lr_rho=1e-2,
          lam_min=1e-6, rho_min=0.0, log_every=10, rng=None,
          loss_and_grad=None):
    """Run projected gradient descent on ``(b, lambda, rho)``.

    Parameters
    ----------
    X : (l, n, d) array
        Training bags.
    w : (l,) array
        Bag labels.
    b_true : (d,) array
        True direction (``||b_true||^2 == d``), used to initialize ``b`` at
        overlap ``u0`` and to track ``u`` during training.
    u0, lam0, rho0 : float
        Initial overlap, inverse temperature, and output scale.
    n_steps : int
        Number of gradient steps.
    lr_b, lr_lam, lr_rho : float
        Learning rates for ``b``, ``lambda``, ``rho``.
    lam_min, rho_min : float
        Lower bounds enforced after each step.
    log_every : int
        Record the trajectory every this many steps (plus step 0).
    rng : numpy.random.Generator, optional
        Used to construct the initial ``b``.
    loss_and_grad : callable, optional
        ``(X, w, b, lam, rho) -> (loss, grad_b, grad_lam, grad_rho)``.
        Defaults to the squared-error loss of :mod:`extremal_mil.model`; pass
        :func:`extremal_mil.logistic.make_loss_and_grad` to train the
        classification variant, with ``w`` then holding the binary labels.

    Returns
    -------
    dict
        History with arrays ``step``, ``u``, ``lam``, ``rho``, ``loss``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if loss_and_grad is None:
        loss_and_grad = mse_loss_and_grad
    d = b_true.shape[0]
    b = make_b0(u0, b_true, rng)
    lam = lam0
    rho = rho0

    hist = {"step": [], "u": [], "lam": [], "rho": [], "loss": []}

    def log(step, loss):
        hist["step"].append(step)
        hist["u"].append(overlap(b, b_true))
        hist["lam"].append(lam)
        hist["rho"].append(rho)
        hist["loss"].append(loss)

    loss0, _, _, _ = loss_and_grad(X, w, b, lam, rho)
    log(0, loss0)

    for step in range(1, n_steps + 1):
        loss, grad_b, grad_lam, grad_rho = loss_and_grad(X, w, b, lam, rho)

        b = b - lr_b * project_tangent(grad_b, b)
        b = np.sqrt(d) * b / np.linalg.norm(b)

        lam = max(lam_min, lam - lr_lam * grad_lam)
        rho = max(rho_min, rho - lr_rho * grad_rho)

        if step % log_every == 0:
            new_loss, _, _, _ = loss_and_grad(X, w, b, lam, rho)
            log(step, new_loss)

    return {k: np.array(v) for k, v in hist.items()}
