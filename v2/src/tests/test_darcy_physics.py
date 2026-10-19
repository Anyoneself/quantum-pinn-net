from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "v2/src/dataset"))

from darcy_physics import (  # noqa: E402
    darcy_flow_loss,
    normalized_log_perm_to_positive_permeability,
    solve_dimensionless_pressure,
)


class DarcyPhysicsTest(unittest.TestCase):
    def test_positive_permeability_and_pressure_boundary_conditions(self) -> None:
        log_perm = torch.zeros(2, 1, 12, 12)
        permeability = normalized_log_perm_to_positive_permeability(log_perm)
        pressure = solve_dimensionless_pressure(permeability, iterations=20)

        self.assertTrue(bool(torch.all(permeability > 0)))
        self.assertTrue(torch.allclose(pressure[:, :, :, 0], torch.ones_like(pressure[:, :, :, 0])))
        self.assertTrue(torch.allclose(pressure[:, :, :, -1], torch.zeros_like(pressure[:, :, :, -1])))
        self.assertTrue(torch.allclose(pressure[:, :, 0, :], pressure[:, :, 1, :]))
        self.assertTrue(torch.allclose(pressure[:, :, -1, :], pressure[:, :, -2, :]))

    def test_uniform_field_satisfies_steady_darcy_equation(self) -> None:
        log_perm = torch.zeros(2, 1, 16, 16, requires_grad=True)
        terms = darcy_flow_loss(log_perm, iterations=12)
        total = terms.continuity + terms.flux_balance
        total.backward()

        self.assertLess(float(terms.continuity.detach()), 1e-8)
        self.assertLess(float(terms.flux_balance.detach()), 1e-8)
        self.assertIsNotNone(log_perm.grad)
        self.assertTrue(bool(torch.isfinite(log_perm.grad).all()))


if __name__ == "__main__":
    unittest.main()
