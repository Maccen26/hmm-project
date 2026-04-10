from unittest import TestCase
import jax.numpy as jnp
import jax
from src.api.v4.parameters.mu_params import MuParameter, MuWorkingParamSet


class TestMuParameter(TestCase):

    def test_construction_valid_scalar(self):
        val = jnp.array([1.0])
        param = MuParameter(val=val)
        self.assertTrue(jnp.allclose(param.val, jnp.array([1.0])))

    def test_construction_rejects_non_jnp_array(self):
        with self.assertRaises(ValueError):
            MuParameter(val=1.0)

    def test_construction_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            MuParameter(val=jnp.array([1.0, 2.0]))

    def test_update_val_returns_new_instance(self):
        param = MuParameter(val=jnp.array([1.0]))
        updated = param.update_val(jnp.array([5.0]))
        self.assertIsInstance(updated, MuParameter)
        self.assertTrue(jnp.allclose(updated.val, jnp.array([5.0])))
        # Original is unchanged
        self.assertTrue(jnp.allclose(param.val, jnp.array([1.0])))


class TestMuWorkingParamSet(TestCase):

    def setUp(self):
        # Natural params: ordered means [1.0, 2.0, 4.0]
        self.natural_params = jnp.array([1.0, 2.0, 4.0])
        # Corresponding working params: [mu0, log(diff1), log(diff2)]
        # = [1.0, log(1.0), log(2.0)] = [1.0, 0.0, log(2.0)]
        self.working_params = jnp.array([1.0, 0.0, jnp.log(2.0)])

    def test_construction_from_working_params(self):
        try:
            ps = MuWorkingParamSet(working_params=self.working_params)
        except Exception as e:
            self.fail(f"Construction failed: {e}")

    def test_to_jnp_array_shape(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        arr = ps.to_jnp_array()
        self.assertEqual(arr.shape, (1, 3))

    def test_to_jnp_array_values(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        arr = ps.to_jnp_array()
        self.assertTrue(jnp.allclose(arr[0], self.working_params))

    def test_get_all_working_params_matches_input(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        wp = ps.get_all_working_params()
        self.assertEqual(wp.shape, (1, 3))
        self.assertTrue(jnp.allclose(wp[0], self.working_params))

    def test_get_all_natural_params_shape(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        natural = ps.get_all_natural_params()
        self.assertEqual(natural.shape, (1, 3))

    def test_from_natural_params_roundtrip(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        ps_from_natural = ps.from_natural_params(self.natural_params)
        recovered_natural = ps_from_natural.get_all_natural_params()
        self.assertTrue(
            jnp.allclose(recovered_natural, self.natural_params, atol=1e-5),
            msg=f"Round-trip failed: expected {self.natural_params}, got {recovered_natural}"
        )

    def test_from_natural_params_working_values(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        ps_from_natural = ps.from_natural_params(self.natural_params)
        wp = ps_from_natural.get_all_working_params()
        self.assertTrue(
            jnp.allclose(wp[0], self.working_params, atol=1e-5),
            msg=f"Expected working params {self.working_params}, got {wp[0]}"
        )

    def test_get_all_natural_params_values(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        natural = ps.get_all_natural_params()
        self.assertTrue(
            jnp.allclose(natural, self.natural_params, atol=1e-5),
            msg=f"Expected {self.natural_params}, got {natural}"
        )

    def test_get_all_natural_params_are_ordered(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        natural = ps.get_all_natural_params()
        diffs = jnp.diff(natural)
        self.assertTrue(jnp.all(diffs > 0), msg="Natural params must be strictly increasing")

    def test_get_working_param_by_index(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        val = ps.get_working_param(index=(0, 0))
        self.assertTrue(jnp.allclose(val, self.working_params[0]))

    def test_get_working_param_last_index(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        val = ps.get_working_param(index=(0, 2))
        self.assertTrue(jnp.allclose(val, self.working_params[2]))

    def test_update_working_param(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        new_val = jnp.array([99.0])
        ps_new = ps.update_working_param(index=(0, 0), new_value=new_val)
        updated_val = ps_new.get_working_param(index=(0, 0))
        self.assertTrue(jnp.allclose(updated_val, new_val))

    def test_get_natural_param_by_index(self):
        # Bug: get_all_natural_params returns 1D array, but get_natural_param indexes with [row, col]
        ps = MuWorkingParamSet(working_params=self.working_params)
        val = ps.get_natural_param(index=(0, 0))
        self.assertTrue(jnp.allclose(val, self.natural_params[0]))

    def test_update_natural_param(self):
        ps = MuWorkingParamSet(working_params=self.working_params)
        new_val = jnp.array([99.0])
        ps_new = ps.update_natural_param(index=(0, 0), new_value=new_val)
        updated_val = ps_new.get_working_param(index=(0, 0))
        self.assertTrue(jnp.allclose(updated_val, new_val))

    def test_two_state_round_trip(self):
        natural_params = jnp.array([0.0, 3.0])
        ps = MuWorkingParamSet(working_params=jnp.array([0.0, jnp.log(3.0)]))
        ps_from_natural = ps.from_natural_params(natural_params)
        recovered = ps_from_natural.get_all_natural_params()
        self.assertTrue(
            jnp.allclose(recovered, natural_params, atol=1e-5),
            msg=f"2-state round-trip failed: expected {natural_params}, got {recovered}"
        )
