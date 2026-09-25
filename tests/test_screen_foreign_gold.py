"""The screen is only worth running if its arithmetic matches the kernel's.

Two of these exist because two real bugs are in the log. `add_as_vote` is E124's
vote bug: rank-meaning a newcomer 50/50 against an already-blended composite
gives it half the ensemble while the comment claims one vote each. And the AUC
has to average ties, because a saturated head that returns the same probability
for many studies otherwise scores by argsort order, which is arbitrary.
"""
import numpy as np
import pytest

from eda.screen_foreign_gold import add_as_vote, auc, macro_auc, paired_bootstrap, rankpct, spearman


def test_rankpct_is_uniform_on_the_unit_interval():
    r = rankpct(np.array([[3.0], [1.0], [2.0], [0.0]]))
    assert r.ravel().tolist() == [1.0, 1 / 3, 2 / 3, 0.0]


def test_auc_is_one_for_a_perfect_ordering_and_half_for_a_constant():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # A constant score is no information, and only tie-averaging says so.
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_auc_is_nan_for_a_single_class_column_and_macro_skips_it():
    assert np.isnan(auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))
    y = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
    s = np.array([[0.1, 0.1], [0.2, 0.9], [0.3, 0.2], [0.4, 0.8]])
    m, per = macro_auc(y, s)
    assert np.isnan(per[0]) and per[1] == 1.0
    assert m == 1.0


def test_add_as_vote_reproduces_the_shipped_kernel_including_its_overcorrection():
    """E127. This asserts what the KERNEL does, which is not what it claims.

    E124 caught the newcomer being given half the blend and weighted the prior
    by how many pipelines it carries. That is the right idea implemented one
    call too far: `rankpct(prior)` RE-UNIFORMISES an average of ranks, undoing
    the variance reduction that made two members count as two. With independent
    uniform inputs the newcomer ends up at 1/sqrt(5) influence where one vote of
    three is 1/sqrt(3). The screen has to carry the same distortion or it would
    be scoring a blend the kernel does not compute.
    """
    rng = np.random.default_rng(0)
    n = 20000
    a, b, c = (rankpct(rng.random((n, 1))) for _ in range(3))
    prior = (a + b) / 2.0
    rho = lambda v: float(np.corrcoef(v.ravel(), c.ravel())[0, 1])  # noqa: E731

    assert rho((a + b + c) / 3.0) == pytest.approx(1 / np.sqrt(3), abs=0.01)
    assert rho(add_as_vote(prior, c, n_prior=2)) == pytest.approx(1 / np.sqrt(5), abs=0.01)
    # the pre-E124 form, for the record: the newcomer at half the ensemble
    assert rho((rankpct(prior) + c) / 2.0) == pytest.approx(1 / np.sqrt(2), abs=0.01)
    # and the one-line correction, which is NOT shipped yet -- it waits for the
    # pending three-package board reading so that it stays a single variable
    assert rho((2 * prior + c) / 3.0) == pytest.approx(1 / np.sqrt(3), abs=0.01)


def test_add_as_vote_is_an_exact_mean_when_the_prior_carries_one_pipeline():
    """n_prior=1 has nothing to re-uniformise, so the 50/50 is honest."""
    rng = np.random.default_rng(3)
    a, c = rankpct(rng.random((200, 2))), rankpct(rng.random((200, 2)))
    assert np.allclose(add_as_vote(a, c, n_prior=1), (a + c) / 2.0)


def test_spearman_is_one_against_a_monotone_transform_of_itself():
    rng = np.random.default_rng(1)
    a = rng.random((50, 4))
    assert spearman(a, np.exp(a))[0] == pytest.approx(1.0)


def test_paired_bootstrap_is_deterministic_and_signs_a_real_gain():
    rng = np.random.default_rng(2)
    y = (rng.random((58, 12)) < 0.35).astype(int)
    noise = rng.normal(0, 1.0, y.shape)
    weak = y + noise
    strong = y + noise * 0.3
    out = paired_bootstrap(y, strong, weak, n_boot=400)
    assert out["delta"] > 0 and out["p_better"] > 0.95
    assert out == paired_bootstrap(y, strong, weak, n_boot=400)
    # and it is symmetric in sign
    back = paired_bootstrap(y, weak, strong, n_boot=400)
    assert back["delta"] == pytest.approx(-out["delta"], abs=1e-12)
