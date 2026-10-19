"""Softmax-pooling student model and its loss/gradients.

Implements eq:model_def_new_param from ``mil.tex``:

.. math::

    \\hat{w}_\\lambda(X^k\\bbeta) := \\rho\\, \\sigma(\\lambda X^k \\bb) \\cdot (X^k \\bb)

where :math:`\\bbeta = \\rho\\bb`, :math:`\\|\\bb\\|^2 = d`, and ``sigma`` is the
softmax over the ``n`` instances of a bag.

The loss is the normalized :math:`\\bar{L} = \\mathbb{E}[(w-\\hat{w}_\\lambda)^2]/d^2`,
so that gradient descent on ``loss_and_grad`` corresponds directly to the
continuous-time flow :math:`\\dot\\theta = -\\partial_\\theta \\bar{L}` of
eq:gradient_flow_def in the small-learning-rate limit.
"""

import numpy as np


def softmax(z, axis=-1):
    """Numerically stable softmax along ``axis``."""
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def what_lambda(X, b, lam, rho):
    """prediction :math:`\\hat{w}_\\lambda(X\\bbeta)` (eq:model_def_new_param).

    Parameters
    ----------
    X : (..., n, d) array
    b : (d,) array, ``||b||^2 == d``
    lam, rho : float

    Returns
    -------
    (...,) array
    """
    z = X @ b
    sigma = softmax(lam * z, axis=-1)
    zbar = (sigma * z).sum(axis=-1)
    return rho * zbar

# HUMAN CHECK
def loss_and_grad(X, w, b, lam, rho):
    """Normalized MSE loss and its gradients w.r.t. ``b``, ``lam``, ``rho``.

    Parameters
    ----------
    X : (l, n, d) array
    w : (l,) array
    b : (d,) array
    lam, rho : float

    Returns
    -------
    loss : float
        :math:`\\bar{L} = \\mathrm{mean}_k(w_k - \\hat{w}_k)^2 / d^2`.
    grad_b : (d,) array
    grad_lam : float
    grad_rho : float
    """
    l, n, d = X.shape

    z = X @ b                                          # (l, n)
    sigma = softmax(lam * z, axis=-1)                  # (l, n)
    zbar = (sigma * z).sum(axis=-1)                    # (l,)
    what = rho * zbar                                  # (l,)
    err = what - w                                     # (l,)

    loss = float(np.mean(err ** 2)) / d ** 2

    # d(what)/d(lambda) = rho * Var_sigma(z)
    var_sigma = (sigma * z ** 2).sum(axis=-1) - zbar ** 2
    dwhat_dlam = rho * var_sigma                       # (l,)

    # d(what)/db = rho * X^T [sigma * (1 + lambda*(z - zbar))]
    gz = sigma * (1.0 + lam * (z - zbar[:, None]))     # (l, n)
    dwhat_db = rho * np.einsum("lnd,ln->ld", X, gz)    # (l, d)

    # d(what)/d(rho) = zbar
    dwhat_drho = zbar                                  # (l,)

    coeff = 2.0 * err / (l * d ** 2)                   # (l,)
    grad_b = coeff @ dwhat_db                          # (d,)
    grad_lam = float(coeff @ dwhat_dlam)
    grad_rho = float(coeff @ dwhat_drho)

    return loss, grad_b, grad_lam, grad_rho
