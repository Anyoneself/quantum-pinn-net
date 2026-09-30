from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen-figures"))
import run_supplement_v2 as module


class SupplementProtocolTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9)
        self.truth = self.rng.normal(size=(4, 32, 32)).astype(np.float32)
        self.train, self.val, self.test = module.masks(self.truth.shape, 0.7, 101)
        self.conditions = np.tile([30, 0.012, 1, 0.22, 120], (4, 1)).astype(np.float32)
        self.conditions[:, 4] += np.arange(4)

    def test_disjoint_and_nested(self):
        low, v, t = module.masks(self.truth.shape, 0.2, 101)
        self.assertTrue(np.array_equal(v, self.val))
        self.assertTrue(np.array_equal(t, self.test))
        self.assertFalse(np.any(low & ~self.train))
        self.assertTrue(np.all(self.train.astype(int) + self.val + self.test == 1))
        train, val, test = module.masks(self.truth.shape, 0.7, 101, True)
        self.assertEqual(int(test.sum()), 4 * 196)
        self.assertFalse(np.any((train | val) & test))

    def test_test_targets_cannot_change_observations_or_prior(self):
        changed = self.truth.copy()
        changed[self.test] = 1e5
        first = module.observed_values(self.truth, self.train, self.val, 0.1, 101)
        second = module.observed_values(changed, self.train, self.val, 0.1, 101)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[self.test] == 0))
        mean1, _ = module.fit_prior(first, self.train, self.conditions)
        first[self.val] = -1e5
        mean2, _ = module.fit_prior(first, self.train, self.conditions)
        np.testing.assert_array_equal(mean1, mean2)

    def test_correction_ignores_hidden_targets(self):
        pred = np.ones_like(self.truth)
        first = module.correct(pred, self.truth, self.train)
        changed = self.truth.copy()
        changed[~self.train] = 1e5
        np.testing.assert_array_equal(first, module.correct(pred, changed, self.train))
        np.testing.assert_allclose(first[self.train], self.truth[self.train])

    def test_interpolators_ignore_hidden_values(self):
        changed = self.truth.copy()
        changed[~self.train] = 1e5
        for kind, params in [("idw", {"k": 8, "power": 2}),
                             ("rbf", {"neighbors": 16, "smoothing": 0.01})]:
            np.testing.assert_array_equal(
                module.interpolate(kind, self.truth, self.train, **params),
                module.interpolate(kind, changed, self.train, **params),
            )

    def test_constant_darcy_field(self):
        pressure, flux, balance, residual = module.flow_response(np.ones((16, 16)) * 2)
        expected = np.broadcast_to(1 - (np.arange(16) + 0.5) / 16, (16, 16))
        np.testing.assert_allclose(pressure, expected, atol=1e-12)
        self.assertAlmostEqual(flux, 1, places=12)
        self.assertLess(balance, 1e-12)
        self.assertLess(residual, 1e-12)

    def test_all_ablations_have_finite_gradients(self):
        torch.set_num_threads(1)
        for method in module.METHODS:
            model = module.make_model(method, 4)
            out = model(torch.randn(32, 31))
            out.square().mean().backward()
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
