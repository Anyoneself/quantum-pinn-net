"""经验NTK的解析、数值和训练无干扰测试。"""

import copy
import unittest

import numpy as np
import torch

from core import (GROUPS, analyze_checkpoint, empirical_jacobian, kernel_statistics,
                  weighted_errors)
from dataset.cases import DataConfig, generate_case, observe
from metrics.physics import LossWeights, components, objective, tensor
from models.baseline_model import BaselineDarcyPINN
from models.darcy_pinn import ModelConfig
from models.qmodel import QDarcyPINN
from qpinn.strategies import StrategyConfig
from qpinn.training import TrainConfig, fit
from run_experiment import probe_indices


class NTKTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data = DataConfig(grid=12, k_train=8, k_validation=4, collocation=4, boundary=2)
        self.case = generate_case(self.data, 411)
        self.obs = observe(self.case, self.data, 101)
        self.config = ModelConfig(features=2, feature_layers=1, hidden=6)

    def test_linear_analytic_kernel_and_unused_parameters(self):
        # 线性核应为XX^T加偏置核，不应丢弃未使用参数列。
        model = torch.nn.Linear(2, 1).double()
        x = tensor([[0.2, 0.3], [0.5, 0.7]])
        unused = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        named = [(f"permeability.encoder.{name}", p) for name, p in model.named_parameters()]
        named.append(("pressure.unused", unused))
        jacobian, columns = empirical_jacobian(model(x), named)
        np.testing.assert_allclose(jacobian @ jacobian.T, x.numpy() @ x.numpy().T + 1)
        np.testing.assert_array_equal(jacobian[:, columns["pressure"]], 0)

    def test_scale_change_is_not_shape_change(self):
        kernel = np.array([[2.0, 0.5], [0.5, 1.0]])
        stats, _ = kernel_statistics(100 * kernel, kernel, np.array([1.0, -1.0]))
        self.assertAlmostEqual(stats["raw_drift"], 99)
        self.assertLess(stats["shape_drift"], 1e-14)
        self.assertIsNone(kernel_statistics(kernel, kernel, np.zeros(2))[0]["direction_alignment"])

    def test_probe_centers_exactly_match_reference(self):
        from dataset.cases import cell_coordinates
        indices = probe_indices(24, 8)
        np.testing.assert_allclose(cell_coordinates(24)[indices], cell_coordinates(8))
        with self.assertRaises(ValueError):
            probe_indices(24, 6)

    def test_weighted_vector_matches_original_loss(self):
        for cls in (BaselineDarcyPINN, QDarcyPINN):
            model = cls(self.config, 101)
            errors = components(model, self.obs.training)
            loss, _ = objective(errors, LossWeights(), torch.ones(self.data.collocation))
            weighted, slices = weighted_errors(model, self.obs.training, LossWeights())
            torch.testing.assert_close(weighted.square().sum(), loss)
            self.assertNotIn("pressure", slices)

    def test_group_sum_psd_and_quantum_mixed_derivative(self):
        # 直接检查加权Darcy残差对量子门的导数，而不是只检查总损失梯度。
        model = QDarcyPINN(self.config, 101)
        indices = probe_indices(12, 4)
        arrays, stats, frequencies, metadata = analyze_checkpoint(
            model, self.obs.training, LossWeights(), self.case.xy[indices], self.case.log_k[indices])
        for kind in ("output", "loss"):
            kernel = arrays[f"{kind}_kernel"]
            np.testing.assert_allclose(kernel, sum(arrays[f"{kind}_kernel_{g}"] for g in GROUPS),
                                       atol=1e-12, rtol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-10)
            self.assertEqual(stats[kind]["raw_drift"], 0)
        self.assertEqual(len(frequencies), 9)
        # 使用真实参数遍历顺序的列映射，读出层可能在特征块之前注册。
        column = metadata["parameter_columns"]["features"][1]
        parameter = model.permeability.features.rotations
        original = float(parameter[0, 0, 1].detach())
        step = 1e-5
        with torch.no_grad():
            parameter[0, 0, 1] = original + step
        plus, _ = weighted_errors(model, self.obs.training, LossWeights())
        plus = plus.detach().numpy()
        with torch.no_grad():
            parameter[0, 0, 1] = original - step
        minus, _ = weighted_errors(model, self.obs.training, LossWeights())
        minus = minus.detach().numpy()
        with torch.no_grad():
            parameter[0, 0, 1] = original
        lo, hi = metadata["loss_slices"]["physics"]
        derivative = arrays["loss_jacobian"][lo:hi, column]
        self.assertGreater(np.linalg.norm(derivative), 1e-8)
        np.testing.assert_allclose(derivative, ((plus - minus) / (2 * step))[lo:hi],
                                   atol=1e-7, rtol=1e-5)

    def test_target_values_do_not_change_kernels(self):
        model = QDarcyPINN(self.config, 101)
        indices = probe_indices(12, 4)
        args = (model, self.obs.training, LossWeights(), self.case.xy[indices])
        first = analyze_checkpoint(*args, self.case.log_k[indices])[0]
        second = analyze_checkpoint(*args, -10 * self.case.log_k[indices])[0]
        for kind in ("output", "loss"):
            np.testing.assert_array_equal(first[f"{kind}_kernel"], second[f"{kind}_kernel"])

    def test_checkpoint_callback_does_not_change_training(self):
        for method in ("classical_plain", "quantum_plain"):
            args = (self.obs.training, self.obs.validation, method, 101, self.config,
                    TrainConfig(epochs=3), StrategyConfig(), LossWeights())
            first = fit(*args)
            snapshots = {}
            second = fit(*args, checkpoint_epochs=[0, 1, 3],
                         checkpoint_callback=lambda epoch, model:
                         snapshots.update({epoch: copy.deepcopy(model.state_dict())}))
            self.assertEqual(first[2], second[2])
            self.assertEqual(first[3], second[3])
            self.assertEqual(set(snapshots), {0, 1, 3})
            for name, value in first[0].state_dict().items():
                torch.testing.assert_close(value, second[0].state_dict()[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
