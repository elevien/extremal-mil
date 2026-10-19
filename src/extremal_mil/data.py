"""Data generation for the multiple-instance extremal-regression model.

Conventions follow ``mil.tex``, Sec. "Data generating process" / "Training":

- Background patterns are iid ``Normal(0, 1)`` entries (mil.tex: "$X^k_0 \\sim
  \\mathrm{Normal}(0, I)$"), so each instance concentrates on the sphere of
  radius :math:`\\sqrt{d}`.
- :math:`\\bb_{\\rm true} = \\sqrt{d}\\, e_1`, so :math:`\\|\\bb_{\\rm true}\\|^2 = d`.
- :math:`\\bbeta_{\\rm true} = \\rho_{\\rm true}\\,\\bb_{\\rm true}`.
- With probability ``q`` the first row of a bag is replaced by
  :math:`\\bbeta_{\\rm true}` (the planted/marked instance).
- The bag label is :math:`w = \\max(X\\bbeta_{\\rm true})` (eq:mlf).
"""

import numpy as np


def make_b_true(d):
    """Return ``b_true = sqrt(d) * e_1`` so that ``||b_true||^2 = d``."""
    b = np.zeros(d)
    b[0] = np.sqrt(d)
    return b


def make_dataset(l, n, d, b_true, rho_true, q, rng):
    """Generate ``l`` bags of ``n`` instances each in ``R^d``.

    Parameters
    ----------
    l : int
        Number of bags.
    n : int
        Number of instances per bag.
    d : int
        Ambient dimension.
    b_true : (d,) array
        Direction of the true coefficients, ``||b_true||^2 == d``.
    rho_true : float
        Magnitude of the true coefficients, ``beta_true = rho_true * b_true``.
    q : float
        Probability that a bag contains the marked instance ``beta_true``.
    rng : numpy.random.Generator

    Returns
    -------
    X : (l, n, d) array
        The bags.
    w : (l,) array
        Bag labels ``w^k = max_i <x_i, beta_true>`` (eq:mlf).
    is_pos : (l,) bool array
        ``True`` for bags that contain the marked instance ``beta_true``.
    """
    beta_true = rho_true * b_true
    X = rng.standard_normal((l, n, d))
    is_pos = rng.random(l) < q
    X[is_pos, 0, :] = beta_true
    y = X @ beta_true
    w = y.max(axis=1)
    return X, w, is_pos
