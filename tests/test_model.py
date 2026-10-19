import numpy as np

from extremal_mil.model import loss_and_grad, softmax, what_lambda


def test_softmax_normalized():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((5, 7))
    s = softmax(z, axis=-1)
    assert np.allclose(s.sum(axis=-1), 1.0)
    assert np.all(s >= 0)


def test_what_lambda_matches_loss_and_grad():
    rng = np.random.default_rng(1)
    l, n, d = 10, 12, 6
    X = rng.standard_normal((l, n, d))
    b = rng.standard_normal(d)
    b = np.sqrt(d) * b / np.linalg.norm(b)
    lam, rho = 0.8, 1.2

    what = what_lambda(X, b, lam, rho)
    w = what + 0.01  # arbitrary labels

    loss, *_ = loss_and_grad(X, w, b, lam, rho)
    expected = np.mean((what - w) ** 2) / d ** 2
    assert np.isclose(loss, expected)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    l, n, d = 30, 15, 6
    X = rng.standard_normal((l, n, d))
    w = rng.standard_normal(l) * 5
    b = rng.standard_normal(d)
    b = np.sqrt(d) * b / np.linalg.norm(b)
    lam, rho = 0.7, 1.3

    _, grad_b, grad_lam, grad_rho = loss_and_grad(X, w, b, lam, rho)

    eps = 1e-6

    loss_p, *_ = loss_and_grad(X, w, b, lam + eps, rho)
    loss_m, *_ = loss_and_grad(X, w, b, lam - eps, rho)
    fd_lam = (loss_p - loss_m) / (2 * eps)
    assert np.isclose(grad_lam, fd_lam, rtol=1e-4, atol=1e-8)

    loss_p, *_ = loss_and_grad(X, w, b, lam, rho + eps)
    loss_m, *_ = loss_and_grad(X, w, b, lam, rho - eps)
    fd_rho = (loss_p - loss_m) / (2 * eps)
    assert np.isclose(grad_rho, fd_rho, rtol=1e-4, atol=1e-8)

    direction = rng.standard_normal(d)
    loss_p, *_ = loss_and_grad(X, w, b + eps * direction, lam, rho)
    loss_m, *_ = loss_and_grad(X, w, b - eps * direction, lam, rho)
    fd_b = (loss_p - loss_m) / (2 * eps)
    assert np.isclose(grad_b @ direction, fd_b, rtol=1e-4, atol=1e-8)
