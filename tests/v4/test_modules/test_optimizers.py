from unittest import TestCase
import jax.numpy as jnp
import jax

from src.api.v4 import HMMParams, StaticTransition, GaussEmission, HMM
from src.api.v4 import GradientSolver, LBFGSSolver, Minimizer


def _make_hmm():
    transition_logits = jnp.array([[-0.84729786, -1.94591015],
                                   [-2.07944154, -1.09861229],
                                   [-1.60943791, -1.09861229]])
    transition = StaticTransition(transition_logits)
    emission = GaussEmission.from_params(
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([1.0, 1.0, 1.0])
    )
    return HMM(transition, emission)


def _make_data(n=50, key=0):
    rng = jax.random.PRNGKey(key)
    return jax.random.normal(rng, shape=(n, 1))


class TestHMMOptimizers(TestCase):

    def setUp(self):
        self.hmm = _make_hmm()
        self.ys = _make_data()

    # ------------------------------------------------------------------
    # Solver stores fitted HMMParams in self.params
    # ------------------------------------------------------------------

    def test_gradient_solver_stores_hmm_params(self):
        solver = GradientSolver(n_iter=5)
        solver.fit(self.hmm.params, self.ys, u_pre=self.hmm.u_pre)
        self.assertIsInstance(solver.params, HMMParams)

    def test_lbfgs_solver_stores_hmm_params(self):
        solver = LBFGSSolver(n_iter=5)
        solver.fit(self.hmm.params, self.ys, u_pre=self.hmm.u_pre)
        self.assertIsInstance(solver.params, HMMParams)

    def test_minimizer_stores_hmm_params(self):
        solver = Minimizer(n_iter=5)
        solver.fit(self.hmm.params, self.ys, u_pre=self.hmm.u_pre)
        self.assertIsInstance(solver.params, HMMParams)

    def test_gradient_solver_fit_returns_none(self):
        solver = GradientSolver(n_iter=5)
        result = solver.fit(self.hmm.params, self.ys, u_pre=self.hmm.u_pre)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Parameter-update tests
    # ------------------------------------------------------------------

    def test_gradient_solver_changes_params(self):
        solver = GradientSolver(n_iter=50)
        original_mu0 = self.hmm.params.emission.mu0
        solver.fit(self.hmm.params, self.ys, u_pre=self.hmm.u_pre)
        self.assertFalse(
            jnp.allclose(solver.params.emission.mu0, original_mu0),
            "mu0 should have changed after 50 gradient steps"
        )

    def test_all_params_trainable_by_default(self):
        solver = GradientSolver(n_iter=20)
        original_mu0 = self.hmm.params.emission.mu0
        original_log_diff = self.hmm.params.emission.log_mu_diff
        solver.fit(self.hmm.params, self.ys, u_pre=self.hmm.u_pre)
        mu0_changed = not jnp.allclose(solver.params.emission.mu0, original_mu0)
        log_diff_changed = not jnp.allclose(solver.params.emission.log_mu_diff, original_log_diff)
        self.assertTrue(mu0_changed or log_diff_changed)

    # ------------------------------------------------------------------
    # HMM.fit() updates self.params in place
    # ------------------------------------------------------------------

    def test_hmm_fit_updates_params_in_place(self):
        original_mu0 = self.hmm.params.emission.mu0
        self.hmm.fit(self.ys, solver=GradientSolver(n_iter=20))
        self.assertFalse(jnp.allclose(self.hmm.params.emission.mu0, original_mu0))

    def test_default_solver_used_when_none(self):
        hmm = _make_hmm()
        hmm.fit(self.ys)  # uses GradientSolver by default
        self.assertIsInstance(hmm.params, HMMParams)

    # ------------------------------------------------------------------
    # Frozen-parameter test
    # ------------------------------------------------------------------

    def test_frozen_parameter_not_updated(self):
        solver = GradientSolver(n_iter=50)
        original_mu0 = self.hmm.params.emission.mu0
        original_log_diff = self.hmm.params.emission.log_mu_diff
        solver.fit(self.hmm.params, self.ys, u_pre=self.hmm.u_pre,
                   frozen={"mu0": False})
        self.assertTrue(
            jnp.allclose(solver.params.emission.mu0, original_mu0),
            "mu0 should be frozen and unchanged"
        )
        self.assertFalse(
            jnp.allclose(solver.params.emission.log_mu_diff, original_log_diff),
            "log_mu_diff should have been updated"
        )

    # ------------------------------------------------------------------
    # Unit test for _build_filter_spec
    # ------------------------------------------------------------------

    def test_filter_spec_frozen_false_means_not_trainable(self):
        import equinox as eqx
        solver = GradientSolver(n_iter=1)
        spec = solver._build_filter_spec(self.hmm.params, frozen={"mu0": False})
        trainable, static = eqx.partition(self.hmm.params, spec)
        self.assertIsNone(trainable.emission.mu0)
        self.assertIsNotNone(static.emission.mu0)

    # ------------------------------------------------------------------
    # Custom loss function test
    # ------------------------------------------------------------------

    def test_custom_loss_fn_is_called(self):
        call_count = {"n": 0}

        def counting_loss(output, params):
            call_count["n"] += 1
            return -output.log_likelihood()

        solver = GradientSolver(n_iter=3)
        solver.fit(self.hmm.params, self.ys, u_pre=self.hmm.u_pre,
                   loss_fn=counting_loss)
        self.assertIsInstance(solver.params, HMMParams)
        self.assertGreater(call_count["n"], 0)
