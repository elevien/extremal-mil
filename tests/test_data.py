import numpy as np

from extremal_mil.data import make_b_true, make_dataset


def test_make_b_true_norm():
    d = 10
    b = make_b_true(d)
    assert np.isclose(np.linalg.norm(b) ** 2, d)
    assert b[0] == np.sqrt(d)
    assert np.allclose(b[1:], 0.0)


def test_make_dataset_shapes_and_labels():
    rng = np.random.default_rng(0)
    l, n, d = 200, 30, 8
    b_true = make_b_true(d)
    rho_true = 1.0

    X, w, is_pos = make_dataset(l, n, d, b_true, rho_true, q=0.5, rng=rng)

    assert X.shape == (l, n, d)
    assert w.shape == (l,)
    assert is_pos.shape == (l,)
    assert is_pos.dtype == bool

    # Roughly half the bags should be positive.
    assert 0.3 < is_pos.mean() < 0.7

    # Positive bags contain beta_true = rho_true * b_true as instance 0, so
    # w >= <beta_true, beta_true> = d * rho_true^2.
    beta_true = rho_true * b_true
    self_overlap = beta_true @ beta_true
    assert np.all(w[is_pos] >= self_overlap - 1e-8)
    assert np.allclose(X[is_pos, 0, :], beta_true)


def test_background_scale():
    """Background instances should have ||x_i|| ~ sqrt(d) (mil.tex convention)."""
    rng = np.random.default_rng(1)
    l, n, d = 500, 50, 20
    b_true = make_b_true(d)
    X, _, _ = make_dataset(l, n, d, b_true, rho_true=1.0, q=0.0, rng=rng)
    norms = np.linalg.norm(X.reshape(-1, d), axis=1)
    assert np.isclose(norms.mean(), np.sqrt(d), rtol=0.1)
