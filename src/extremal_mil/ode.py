"""Gradient-flow ODE on (lambda, u, rho).

Implements the continuous-time flow of ``mil.tex``, Sec. "Gradient flow on
(lambda, u, rho)":

.. math::

    \\dot\\lambda = -\\partial_\\lambda L, \\qquad
    \\dot u = (1-u^2)(-\\partial_u L), \\qquad
    \\dot\\rho = -\\partial_\\rho L,

where, by the master gradient rule (eq:master_gradient_rule),

.. math::

    -\\partial_x L = 2\\Bigl[q\\,\\Delta^+\\,\\partial_x(\\hat{w}^+/d)
        + (1-q)\\,\\Delta^-\\,\\partial_x(\\hat{w}^-/d)\\Bigr],
    \\quad x \\in \\{\\lambda, u, \\rho\\},

with residuals :math:`\\Delta^\\pm = w^{k,\\pm}/d - \\hat{w}^\\pm/d`,
:math:`w^{k,+}/d = \\rho_{\\rm true}^2`, :math:`w^{k,-}/d =
\\sqrt{2\\alpha}\\,\\rho_{\\rm true}`, and the Jacobians of eq:jacobian_table.

The general implementation below (:func:`gradient_flow_rhs`) automatically
reduces to the explicit HR/LR/HN/LN blocks written out at the end of that
section -- this is checked in ``tests/test_ode.py``.
"""

from scipy.integrate import solve_ivp

from .theory import is_high_temp, is_retrieval, lambda_star


def _what_over_d_and_jac(form, lam, u, rho, alpha, rho_true):
    """Value and Jacobian ``(d/dlam, d/du, d/drho)`` of one of the three
    functional forms appearing in Table tab:phases (eq:jacobian_table).

    ``form`` is one of:

    - ``'R'``: :math:`\\hat{w}/d = \\rho\\rho_{\\rm true}u`
    - ``'H'``: :math:`\\hat{w}/d = \\lambda\\rho`
    - ``'L'``: :math:`\\hat{w}/d = \\sqrt{2\\alpha}\\,\\rho`
    """
    if form == "R":
        value = rho * rho_true * u
        jac = (0.0, rho * rho_true, rho_true * u)
    elif form == "H":
        value = lam * rho
        jac = (rho, 0.0, lam)
    elif form == "L":
        lam_s = lambda_star(alpha)
        value = lam_s * rho
        jac = (0.0, 0.0, lam_s)
    else:
        raise ValueError(f"unknown form {form!r}")
    return value, jac


def gradient_flow_rhs(t, state, alpha, rho_true, q):
    """Right-hand side ``d(lambda, u, rho)/dt`` for :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    t : float
        Time (unused; the flow is autonomous).
    state : sequence of 3 floats
        ``(lambda, u, rho)``.
    alpha, rho_true, q : float
        Problem parameters.

    Returns
    -------
    list of 3 floats
        ``(dlambda/dt, du/dt, drho/dt)``.
    """
    lam, u, rho = state

    retrieval = is_retrieval(lam, u, rho_true, alpha)
    high_temp = is_high_temp(lam, alpha)

    form_plus = "R" if retrieval else ("H" if high_temp else "L")
    form_minus = "H" if high_temp else "L"

    what_plus, jac_plus = _what_over_d_and_jac(form_plus, lam, u, rho, alpha, rho_true)
    what_minus, jac_minus = _what_over_d_and_jac(form_minus, lam, u, rho, alpha, rho_true)

    delta_plus = rho_true ** 2 - what_plus
    delta_minus = lambda_star(alpha) * rho_true - what_minus

    def neg_dL(idx):
        return 2.0 * (q * delta_plus * jac_plus[idx] + (1 - q) * delta_minus * jac_minus[idx])

    dlam = neg_dL(0)
    du = (1.0 - u ** 2) * neg_dL(1)
    drho = neg_dL(2)
    return [dlam, du, drho]


def integrate(ic, t_span, alpha, rho_true, q, **kwargs):
    """Integrate the gradient flow from initial condition ``ic = (lambda0, u0, rho0)``.

    Extra keyword arguments are passed through to
    :func:`scipy.integrate.solve_ivp` (e.g. ``t_eval``, ``max_step``,
    ``method``).

    Returns
    -------
    scipy.integrate.OdeResult
    """
    kwargs.setdefault("max_step", 0.01)
    kwargs.setdefault("method", "RK45")
    return solve_ivp(gradient_flow_rhs, t_span, ic, args=(alpha, rho_true, q), **kwargs)
