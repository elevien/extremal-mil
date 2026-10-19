import numpy as np

from extremal_mil.theory import (
    classify_phase,
    is_high_temp,
    is_retrieval,
    lambda_star,
    loss_asymp,
    phi_alpha,
    residuals,
    what_over_d,
)


def test_lambda_star():
    assert np.isclose(lambda_star(0.5), 1.0)
    assert np.isclose(lambda_star(0.125), 0.5)


def test_phi_alpha_continuous_at_lambda_star():
    alpha = 0.4
    ls = lambda_star(alpha)
    left = phi_alpha(ls - 1e-9, alpha)
    right = phi_alpha(ls + 1e-9, alpha)
    assert np.isclose(left, right, atol=1e-6)
    assert np.isclose(right, np.sqrt(2 * alpha))


def test_classify_phase_all_four_reachable():
    alpha, rho_true = 0.3, 1.0
    assert classify_phase(0.5, 0.95, alpha, rho_true) == "HR"
    assert classify_phase(1.0, 0.9, alpha, rho_true) == "LR"
    assert classify_phase(0.5, 0.3, alpha, rho_true) == "HN"
    assert classify_phase(1.0, 0.3, alpha, rho_true) == "LN"


def test_what_over_d_at_perfect_recovery():
    """At u=1, rho=rho_true, lambda=lambda_star, predictions should match the
    true bag outputs (Delta's vanish in HR/LR boundary)."""
    alpha, rho_true = 0.3, 1.0
    lam = lambda_star(alpha)
    u, rho = 1.0, rho_true

    what_plus, what_minus = what_over_d(lam, u, rho, alpha, rho_true)

    assert np.isclose(what_plus, rho_true ** 2)  # matches w^{k,+}/d
    assert np.isclose(what_minus, np.sqrt(2 * alpha) * rho_true)  # matches w^{k,-}/d

    loss = loss_asymp(lam, u, rho, alpha, rho_true, q=0.5)
    assert np.isclose(loss, 0.0, atol=1e-12)


def test_residuals_keys():
    res = residuals(0.5, 0.5, 1.0, 0.3, 1.0)
    assert set(res.keys()) == {"R", "HN", "LN", "H", "L"}


def test_high_temp_and_retrieval_consistency():
    alpha, rho_true = 0.2, 1.0
    lam = lambda_star(alpha)
    assert not is_high_temp(lam, alpha)  # lambda == lambda* is not < lambda*
    assert is_high_temp(lam - 1e-9, alpha)
    # phi_alpha is continuous, so retrieval boundary is also continuous
    u_c = phi_alpha(lam, alpha) / rho_true
    assert is_retrieval(lam, u_c + 1e-6, rho_true, alpha)
    assert not is_retrieval(lam, u_c - 1e-6, rho_true, alpha)
