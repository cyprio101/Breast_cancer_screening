import numpy as np

import mammo_cbm as M


def _synthetic(n, rng):
    y = rng.integers(0, 2, n)
    p = np.clip(rng.normal(0.25 + 0.5 * y, 0.2), 0.001, 0.999)
    return p, y


def test_class_conditional_coverage():
    rng = np.random.default_rng(0)
    pc, yc = _synthetic(4000, rng)
    pt, yt = _synthetic(20000, rng)
    q = M.conformal_thresholds(pc, yc, alpha=0.10)
    in_set = np.where(yt[:, None] == 1, (1 - pt)[:, None], pt[:, None]).ravel() <= q[yt]
    for k in (0, 1):
        assert in_set[yt == k].mean() >= 0.88  # nominal 0.90, small finite-sample slack


def test_verdicts():
    q = np.array([0.5, 0.5])
    assert M.verdict_from_p(0.95, q) == "Suspicious"
    assert M.verdict_from_p(0.05, q) == "Likely benign"
    assert M.verdict_from_p(0.50, q) == "Uncertain"
    assert M.verdict_from_p(0.50, np.array([0.1, 0.1])) == "Uncertain"  # empty set also means refer
