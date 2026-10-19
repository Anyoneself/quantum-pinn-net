from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PROJECT_ROOT / "v2/src/experiment/run_pinn_experiment.py"
spec = importlib.util.spec_from_file_location("run_pinn_experiment_protocol", MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ExperimentProtocolTest(unittest.TestCase):
    def test_three_way_spatial_masks_are_disjoint_and_complete(self) -> None:
        fields = np.zeros((8, 16, 16), dtype=np.float32)
        train, validation, test = module.make_spatial_masks(fields, 0.70, 0.10, seed=42)
        total = train + validation + test

        self.assertTrue(np.all(total == 1.0))
        self.assertFalse(np.any(np.logical_and(train > 0, validation > 0)))
        self.assertFalse(np.any(np.logical_and(train > 0, test > 0)))
        self.assertFalse(np.any(np.logical_and(validation > 0, test > 0)))
        self.assertAlmostEqual(float(train.mean()), 0.70, delta=0.04)
        self.assertAlmostEqual(float(validation.mean()), 0.10, delta=0.03)
        self.assertAlmostEqual(float(test.mean()), 0.20, delta=0.04)

    def test_dataset_exposes_distinct_validation_and_test_masks(self) -> None:
        fields = np.zeros((2, 4, 4), dtype=np.float32)
        conditions = np.zeros((2, 5), dtype=np.float32)
        expected = np.zeros(2, dtype=np.float32)
        train, validation, test = module.make_spatial_masks(fields, 0.60, 0.20, seed=7)
        dataset = module.FieldDataset(
            fields,
            conditions,
            conditions,
            expected,
            train,
            validation,
            test,
        )

        item = dataset[0]
        self.assertIn("validation_mask", item)
        self.assertIn("test_mask", item)
        self.assertFalse(np.array_equal(item["validation_mask"].numpy(), item["test_mask"].numpy()))


if __name__ == "__main__":
    unittest.main()
