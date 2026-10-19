"""Asymptotic predictions from ``mil.tex``, Sec. "Asymptotic of the average loss".

This module implements:

- :func:`phi_alpha`, :func:`lambda_star` -- the REM free-energy density and its
  critical inverse temperature (eq:noise_free_energy).
- :func:`is_retrieval`, :func:`is_high_temp`, :func:`classify_phase` -- the
  retrieval criterion (eq:retrieval_criterion_general) and the four phases
  HR/LR/HN/LN of Table tab:phases.
- :func:`residuals` -- the five residual shorthands (eq:residual_defs).
- :func:`what_over_d` -- the asymptotic predictions :math:`\\hat{w}^\\pm/d`
  (eq:wlam_cases / Table tab:phases).
- :func:`loss_asymp` -- the asymptotic loss :math:`L = q\\bar{L}_+ +
  (1-q)\\bar{L}_-` (Table tab:phases).
"""

import numpy as np


def lambda_star(alpha):
    """Critical inverse temperature :math:`\\lambda^* = \\sqrt{2\\alpha}`."""
    return np.sqrt(2.0 * alpha)


def phi_alpha(lam, alpha):
    """REM free-energy density :math:`\\phi_\\alpha(\\lambda)` (eq:noise_free_energy)."""
    if lam < lambda_star(alpha):
        return alpha / lam + lam / 2.0
    return np.sqrt(2.0 * alpha)


def is_retrieval(lam, u, rho_true, alpha):
    """Retrieval criterion :math:`\\rho_{\\rm true} u > \\phi_\\alpha(\\lambda)`
    (eq:retrieval_criterion_general)."""
    return rho_true * u > phi_alpha(lam, alpha)


def is_high_temp(lam, alpha):
    """High-temperature (paramagnetic) phase: :math:`\\lambda < \\lambda^*`."""
    return lam < lambda_star(alpha)


def classify_phase(lam, u, alpha, rho_true):
    """Classify ``(lambda, u)`` into one of the four phases of Table tab:phases.

    Returns one of ``'HR'``, ``'LR'``, ``'HN'``, ``'LN'``.
    """
    retrieval = is_retrieval(lam, u, rho_true, alpha)
    high_temp = is_high_temp(lam, alpha)
    if high_temp and retrieval:
        return "HR"
    if (not high_temp) and retrieval:
        return "LR"
    if high_temp and not retrieval:
        return "HN"
    return "LN"


def residuals(lam, u, rho, alpha, rho_true):
    """The five residual shorthands of eq:residual_defs."""
    lam_s = lambda_star(alpha)
    return {
        "R": rho_true - rho * u,
        "HN": rho_true ** 2 - lam * rho,
        "LN": rho_true ** 2 - lam_s * rho,
        "H": lam_s * rho_true - lam * rho,
        "L": rho_true - rho,
    }


def what_over_d(lam, u, rho, alpha, rho_true):
    """Asymptotic predictions :math:`(\\hat{w}^+/d, \\hat{w}^-/d)` (Table tab:phases)."""
    lam_s = lambda_star(alpha)
    retrieval = is_retrieval(lam, u, rho_true, alpha)
    high_temp = is_high_temp(lam, alpha)

    if retrieval:
        what_plus = rho * rho_true * u
    elif high_temp:
        what_plus = lam * rho
    else:
        what_plus = lam_s * rho

    what_minus = lam * rho if high_temp else lam_s * rho

    return what_plus, what_minus


def loss_asymp(lam, u, rho, alpha, rho_true, q):
    """Asymptotic loss :math:`L = q\\bar{L}_+ + (1-q)\\bar{L}_-` (Table tab:phases)."""
    phase = classify_phase(lam, u, alpha, rho_true)
    res = residuals(lam, u, rho, alpha, rho_true)

    if phase == "HR":
        L_plus = rho_true ** 2 * res["R"] ** 2
        L_minus = res["H"] ** 2
    elif phase == "LR":
        L_plus = rho_true ** 2 * res["R"] ** 2
        L_minus = 2 * alpha * res["L"] ** 2
    elif phase == "HN":
        L_plus = res["HN"] ** 2
        L_minus = res["H"] ** 2
    else:  # LN
        L_plus = res["LN"] ** 2
        L_minus = 2 * alpha * res["L"] ** 2

    return q * L_plus + (1 - q) * L_minus
