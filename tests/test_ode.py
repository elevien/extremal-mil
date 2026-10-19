"""Check that the general gradient-flow rhs reproduces the explicit HR/LR/HN/LN
ODE blocks written out at the end of mil.tex, Sec. "Gradient flow on
(lambda, u, rho)"."""

import numpy as np

from extremal_mil.ode import gradient_flow_rhs
from extremal_mil.theory import classify_phase, lambda_star, residuals

ALPHA = 0.3
RHO_TRUE = 1.0
Q = 0.4

# (lambda, u, rho) chosen so that each falls in a distinct phase.
CASES = {
    "HR": (0.5, 0.95, 0.8),
    "LR": (1.0, 0.9, 0.8),
    "HN": (0.5, 0.3, 0.8),
    "LN": (1.0, 0.3, 0.8),
}


def explicit_rhs(phase, lam, u, rho):
    res = residuals(lam, u, rho, ALPHA, RHO_TRUE)
    lam_s = lambda_star(ALPHA)

    if phase == "HR":
        dlam = 2 * (1 - Q) * rho * res["H"]
        du = 2 * Q * rho * RHO_TRUE ** 2 * res["R"] * (1 - u ** 2)
        drho = 2 * Q * u * RHO_TRUE ** 2 * res["R"] + 2 * (1 - Q) * lam * res["H"]
    elif phase == "LR":
        dlam = 0.0
        du = 2 * Q * rho * RHO_TRUE ** 2 * res["R"] * (1 - u ** 2)
        drho = 2 * Q * u * RHO_TRUE ** 2 * res["R"] + 4 * (1 - Q) * ALPHA * res["L"]
    elif phase == "HN":
        common = Q * res["HN"] + (1 - Q) * res["H"]
        dlam = 2 * rho * common
        du = 0.0
        drho = 2 * lam * common
    else:  # LN
        dlam = 0.0
        du = 0.0
        drho = 2 * Q * lam_s * res["LN"] + 4 * (1 - Q) * ALPHA * res["L"]

    return [dlam, du, drho]


def test_gradient_flow_matches_explicit_phase_blocks():
    for phase, (lam, u, rho) in CASES.items():
        assert classify_phase(lam, u, ALPHA, RHO_TRUE) == phase
        general = gradient_flow_rhs(0.0, [lam, u, rho], ALPHA, RHO_TRUE, Q)
        explicit = explicit_rhs(phase, lam, u, rho)
        np.testing.assert_allclose(general, explicit, rtol=1e-12)


def test_du_vanishes_at_u_equal_one():
    """The Riemannian factor (1-u^2) should kill du/dt at u=+-1."""
    for phase, (lam, _, rho) in CASES.items():
        for u in (1.0, -1.0):
            _, du, _ = gradient_flow_rhs(0.0, [lam, u, rho], ALPHA, RHO_TRUE, Q)
            assert du == 0.0


def test_integrate_runs():
    from extremal_mil.ode import integrate

    sol = integrate((0.5, 0.2, 0.5), (0.0, 1.0), alpha=0.3, rho_true=1.0, q=0.5)
    assert sol.success
    assert sol.y.shape[0] == 3
