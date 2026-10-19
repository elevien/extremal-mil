"""Check that the parallel sweep driver reproduces plain serial ``gd.train``.

The point of ``extremal_mil.sweeps`` is purely to run many ``gd.train`` calls
over one shared dataset across processes; it must not change the numbers.
"""

import numpy as np

from extremal_mil import data, gd, sweeps

ALPHA = 0.05
RHO_TRUE = 1.2
Q = 0.2
LAM0 = 0.2
RHO0 = 1.7
N_STEPS = 12
LR = 1e-1

INITS = [(LAM0, u0, RHO0) for u0 in (-0.6, 0.0, 0.3, 0.8)]


def test_n_from_d():
    # n = e^{alpha d}, rounded.
    assert sweeps.n_from_d(90, ALPHA) == int(round(np.exp(0.05 * 90)))
    assert sweeps.n_from_d(0, ALPHA) == 1


def test_gd_finals_matches_serial_gd_train():
    d = 12
    n = sweeps.n_from_d(d, ALPHA)
    b_true = data.make_b_true(d)
    X, w, _ = data.make_dataset(40, n, d, b_true, RHO_TRUE, Q,
                                np.random.default_rng(0))

    expected = np.array([
        [h["lam"][-1], h["u"][-1], h["rho"][-1]]
        for h in (
            gd.train(X, w, b_true, u0=u0, lam0=lam0, rho0=rho0,
                     n_steps=N_STEPS, lr_b=LR, lr_lam=LR, lr_rho=LR,
                     log_every=N_STEPS, rng=np.random.default_rng(1))
            for lam0, u0, rho0 in INITS
        )
    ])

    for n_workers in (1, 3):
        got = sweeps.gd_finals(X, w, b_true, INITS, N_STEPS, LR,
                               b0_seed=1, n_workers=n_workers)
        stacked = np.column_stack([got["lam"], got["u"], got["rho"]])
        assert np.array_equal(stacked, expected)

    # The shared-dataset global is not left populated after a sweep.
    assert not sweeps._SHARED


def test_sweep_dataset_matches_manual_dataset():
    d = 12
    kwargs = dict(n_steps=N_STEPS, lr=LR, b0_seed=1, n_workers=1)
    res, n = sweeps.sweep_dataset(40, d, ALPHA, RHO_TRUE, Q, INITS,
                                  data_seed=7, **kwargs)
    assert n == sweeps.n_from_d(d, ALPHA)

    b_true = data.make_b_true(d)
    X, w, _ = data.make_dataset(40, n, d, b_true, RHO_TRUE, Q,
                                np.random.default_rng(7))
    ref = sweeps.gd_finals(X, w, b_true, INITS, N_STEPS, LR,
                           b0_seed=1, n_workers=1)
    for key in ("lam", "u", "rho"):
        assert np.array_equal(res[key], ref[key])


def test_theory_finals_matches_direct_integration():
    from extremal_mil import ode

    inits = [(LAM0, 0.1, RHO0), (LAM0, 0.8, RHO0)]
    got = sweeps.theory_finals(inits, ALPHA, RHO_TRUE, Q, t_max=50.0, max_step=2.0)
    for i, ic in enumerate(inits):
        sol = ode.integrate(list(ic), [0, 50.0], ALPHA, RHO_TRUE, Q, max_step=2.0)
        assert np.allclose([got["lam"][i], got["u"][i], got["rho"][i]], sol.y[:, -1])
