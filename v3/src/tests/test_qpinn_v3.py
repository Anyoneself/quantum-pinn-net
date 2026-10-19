from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from dataset.cases import (DataConfig, ReferenceCase, ValidationView, cell_coordinates,
                        field_parameters, generate_case, gstools_log_field, log_field,
                        observe, solve_darcy)
from experiment.runner import run_experiment, verify_experiment
from models.baseline_model import BaselineDarcyPINN, ClassicalFeatures
from models.qmodel import QDarcyPINN
from models.quantum_pinn import DarcyPINN, ModelConfig, QuantumFeatures
from metrics.physics import (LossWeights, components, darcy_residual, objective,
                           tensor, validation_score)
from qpinn.strategies import ResidualAttention, StrategyConfig, screen_features
from qpinn.training import METHODS, TrainConfig, fit


def small_data():
    return DataConfig(grid=8, k_train=8, k_validation=4,
                      collocation=8, boundary=4, modes=2)


class DataProtocolTests(unittest.TestCase):
    def test_uniform_finite_volume_solution(self):
        solved = solve_darcy(np.full((8, 12), 2.5))
        expected = np.broadcast_to(1 - (np.arange(12) + 0.5) / 12, (8, 12))
        np.testing.assert_allclose(solved["pressure"], expected, atol=1e-13)
        self.assertAlmostEqual(solved["inlet"], 2.5, places=12)
        self.assertLess(solved["balance"], 1e-12)
        self.assertLess(solved["solver_residual"], 1e-12)

    def test_nonuniform_finite_volume_solution_converges_to_analytic_pressure(self):
        errors = []
        for grid in (16, 32, 64):
            x = (np.arange(grid) + 0.5) / grid
            permeability = np.broadcast_to(np.exp(0.7 * x), (grid, grid))
            solved = solve_darcy(permeability)
            exact = (np.exp(-0.7 * x) - np.exp(-0.7)) / (1 - np.exp(-0.7))
            errors.append(float(np.max(np.abs(solved["pressure"] - exact))))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 3.5)
            self.assertLess(coarse / fine, 4.5)
        self.assertLess(errors[2], 1e-4)
        self.assertAlmostEqual(solved["inlet"], 0.7 / (1 - np.exp(-0.7)), delta=1e-3)

    def test_source_is_reproducible_without_field_normalization(self):
        config = small_data()
        first, second = generate_case(config, 11), generate_case(config, 11)
        np.testing.assert_array_equal(first.log_k, second.log_k)
        np.testing.assert_array_equal(first.pressure, second.pressure)
        self.assertEqual(first.metadata, second.metadata)
        self.assertTrue(first.metadata["generator"]["normalization"].startswith("none"))
        self.assertLess(first.metadata["fine_solver_residual"], 1e-11)
        third = generate_case(config, 22)
        self.assertFalse(np.allclose(first.log_k, third.log_k))

    def test_hidden_targets_do_not_affect_observations(self):
        config = replace(small_data(), noise=0.1)
        case = generate_case(config, 11)
        observations = observe(case, config, 101)
        altered = ReferenceCase(case.xy.copy(), case.log_k.copy(), case.pressure.copy(), case.metadata)
        altered.log_k[observations.k_test_indices] = 1e5
        altered.pressure[observations.p_test_indices] = -1e5
        again = observe(altered, config, 101)
        for view in ("training", "validation"):
            for name, value in vars(getattr(observations, view)).items():
                np.testing.assert_array_equal(value, getattr(getattr(again, view), name))
        altered.log_k[observations.k_validation_indices] = -1e5
        altered.pressure[observations.p_validation_indices] = 1e5
        again = observe(altered, config, 101)
        for name, value in vars(observations.training).items():
            np.testing.assert_array_equal(value, getattr(again.training, name))

    def test_splits_partition_each_target(self):
        config = small_data()
        observations = observe(generate_case(config, 11), config, 101)
        for target in ("k", "p"):
            indices = np.concatenate([getattr(observations, f"{target}_{part}_indices")
                                      for part in ("train", "validation", "test")])
            np.testing.assert_array_equal(np.sort(indices), np.arange(config.grid**2))
        self.assertFalse(hasattr(observations.training, "k_test_indices"))
        self.assertFalse(hasattr(observations.training, "reference"))

    def test_bad_data_configuration_is_rejected(self):
        for config in (replace(small_data(), refinement=2), replace(small_data(), k_train=64),
                       replace(small_data(), noise=-1), replace(small_data(), log_amplitude=float("nan")),
                       replace(small_data(), p_train=8), replace(small_data(), log_variance=-1),
                       replace(small_data(), correlation_length=0), replace(small_data(), anisotropy=0),
                       replace(small_data(), angle=float("nan")), replace(small_data(), gstools_modes=0),
                       replace(small_data(), field_generator="unknown")):
            with self.assertRaises(ValueError):
                generate_case(config, 11)

    def test_gstools_is_default_and_grids_share_one_realization(self):
        config = small_data()
        case = generate_case(config, 11)
        direct, metadata = gstools_log_field(cell_coordinates(config.grid), config, 11)
        np.testing.assert_allclose(case.log_k, direct, atol=1e-13, rtol=1e-13)
        self.assertEqual(metadata["library"], "gstools")
        self.assertEqual(metadata["covariance"], "Gaussian")
        self.assertFalse(metadata["v2_arrays_reused"])
        self.assertEqual(case.metadata["observed_variables"], ["log_k"])
        self.assertEqual(case.metadata["reference_pressure_role"], "evaluation_only")

    def test_gstools_preserves_configured_mean_without_sample_normalization(self):
        config = small_data()
        first = generate_case(config, 11)
        shifted = generate_case(replace(config, log_mean=0.7), 11)
        np.testing.assert_allclose(shifted.log_k - first.log_k, 0.7, atol=1e-13)

    def test_legacy_generator_is_explicit_and_unchanged(self):
        config = replace(small_data(), field_generator="trigonometric")
        case = generate_case(config, 11)
        expected = log_field(case.xy, field_parameters(11, config.modes, config.log_amplitude))
        np.testing.assert_array_equal(case.log_k, expected)
        self.assertEqual(case.metadata["generator"]["library"], "legacy-trigonometric")

    def test_pressure_reference_cannot_affect_permeability_only_training(self):
        config = small_data()
        case = generate_case(config, 11)
        original = observe(case, config, 101)
        poisoned = replace(case, pressure=np.full_like(case.pressure, np.nan))
        altered = observe(poisoned, config, 101)
        for view in ("training", "validation"):
            for name, value in vars(getattr(original, view)).items():
                np.testing.assert_array_equal(value, getattr(getattr(altered, view), name))
        self.assertEqual(original.training.pressure.size, 0)
        self.assertEqual(original.validation.pressure.size, 0)
        self.assertEqual(len(original.p_test_indices), config.grid**2)
        self.assertEqual(original.metadata["noise_pressure_scale_training_only"], 0)
        for kind in ("classical", "quantum"):
            common = (kind + "_rba_mc", 101, ModelConfig(features=2, hidden=6, feature_layers=1),
                      TrainConfig(epochs=2), StrategyConfig(warmup=0, mc_every=1, min_features=1,
                                                         mc_trials=2, mc_batch=4), LossWeights())
            first = fit(original.training, original.validation, *common)
            second = fit(altered.training, altered.validation, *common)
            self.assertEqual(first[2], second[2])
            self.assertEqual(first[3], second[3])
            for a, b in zip(first[0].state_dict().values(), second[0].state_dict().values()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_pressure_supervision_requires_explicit_counts(self):
        config = replace(small_data(), p_train=8, p_validation=4)
        observations = observe(generate_case(config, 11), config, 101)
        self.assertEqual(len(observations.training.pressure), 8)
        self.assertEqual(len(observations.validation.pressure), 4)
        self.assertEqual(observations.metadata["observed_variables"], ["log_k", "pressure"])

    def test_cli_defaults_to_gstools_without_pressure_labels(self):
        from qpinn.cli import parser
        args = parser().parse_args(["run", "--output", "unused"])
        self.assertEqual(args.field_generator, "gstools")
        self.assertEqual(args.p_train, 0)


class CircuitPhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_quantum_expectations_match_known_two_qubit_state(self):
        block = QuantumFeatures(2, 1)
        with torch.no_grad():
            block.rotations.zero_()
        angles = tensor([[0.2, 0.4], [0.5, -0.7]])
        actual = block(angles)
        expected = torch.stack((angles[:, 0].cos(), angles[:, 0].cos() * angles[:, 1].cos()), -1)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.all(actual.abs() <= 1))

    def test_shared_layers_and_pressure_initialization_match(self):
        config = ModelConfig()
        classical, quantum = BaselineDarcyPINN(config, 101), QDarcyPINN(config, 101)
        self.assertIsInstance(classical.permeability.features, ClassicalFeatures)
        self.assertIsInstance(quantum.permeability.features, QuantumFeatures)
        for kind, expected in (("classical", classical), ("quantum", quantum)):
            legacy = DarcyPINN(config, kind, 101)
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(value, legacy.state_dict()[name])
        for name in ("encoder", "readout"):
            for a, b in zip(getattr(classical.permeability, name).parameters(),
                            getattr(quantum.permeability, name).parameters()):
                torch.testing.assert_close(a, b)
        for a, b in zip(classical.pressure.parameters(), quantum.pressure.parameters()):
            torch.testing.assert_close(a, b)
        xy = tensor([[0, 0.3], [1, 0.7]])
        torch.testing.assert_close(classical.pressure(xy), tensor([1, 0]))

    def test_quantum_pde_gradient_matches_finite_difference(self):
        model = DarcyPINN(ModelConfig(features=2, feature_layers=1, hidden=6), "quantum", 101)
        xy = tensor([[0.2, 0.3], [0.7, 0.6]])
        loss = darcy_residual(model, xy).square().mean()
        loss.backward()
        weights = model.permeability.features.rotations
        self.assertIsNotNone(weights.grad)
        self.assertTrue(torch.isfinite(weights.grad).all())
        index = np.unravel_index(weights.grad.abs().argmax().item(), tuple(weights.shape))
        analytic = float(weights.grad[index])
        self.assertGreater(abs(analytic), 1e-9)
        original = float(weights[index].detach())
        step = 1e-5
        with torch.no_grad():
            weights[index] = original + step
        plus = float(darcy_residual(model, xy).square().mean().detach())
        with torch.no_grad():
            weights[index] = original - step
        minus = float(darcy_residual(model, xy).square().mean().detach())
        with torch.no_grad():
            weights[index] = original
        self.assertAlmostEqual(analytic, (plus - minus) / (2 * step), delta=1e-6)

    def test_nonuniform_manufactured_solution_has_zero_pde_residual(self):
        class KnownPermeability(nn.Module):
            def forward(self, xy, mask=None):
                return 0.7 * xy[:, 0]

        class KnownPressure(nn.Module):
            def forward(self, xy):
                return (torch.exp(-0.7 * xy[:, 0]) - np.exp(-0.7)) / (1 - np.exp(-0.7))

        model = nn.Module()
        model.permeability = KnownPermeability()
        model.pressure = KnownPressure()
        result = darcy_residual(model, tensor([[0.2, 0.4], [0.7, 0.8]]))
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-12, rtol=0)


class StrategyTests(unittest.TestCase):
    def test_rba_bounded_and_scale_invariant(self):
        config = StrategyConfig()
        first, second = ResidualAttention(3, config), ResidualAttention(3, config)
        for _ in range(100):
            a = first.update(tensor([0.0, 1.0, 2.0]))
            b = second.update(tensor([0.0, 10.0, 20.0]))
        torch.testing.assert_close(a, b)
        self.assertEqual(float(a[0]), 1.0)
        self.assertGreater(float(a[2]), float(a[1]))
        self.assertLessEqual(float(a.max()), 1 + config.rba_gain)
        self.assertFalse(a.requires_grad)

    def test_mc_is_paired_reproducible_and_removes_unused_feature(self):
        data = small_data()
        observations = observe(generate_case(data, 11), data, 101)
        config = ModelConfig(features=4, hidden=6, feature_layers=1)
        model = DarcyPINN(config, "quantum", 101)
        with torch.no_grad():
            model.permeability.readout[0].weight.zero_()
        strategy = StrategyConfig(mc_trials=2, mc_batch=4, min_features=3, mc_threshold=0)
        event = screen_features(model, observations.training, observations.validation, LossWeights(),
                                torch.ones(data.collocation, dtype=torch.float64), strategy,
                                np.random.default_rng(19))
        self.assertTrue(event["accepted"])
        self.assertEqual(event["candidate_features"], [0])
        self.assertEqual(float(model.permeability.feature_mask[0]), 0)
        self.assertLessEqual(event["validation_after"], event["validation_before"])
        second = screen_features(model, observations.training, observations.validation, LossWeights(),
                                 torch.ones(data.collocation, dtype=torch.float64), strategy,
                                 np.random.default_rng(19))
        self.assertEqual(second["reason"], "minimum_features")

    def test_physics_loss_has_gradients_for_both_networks(self):
        config = small_data()
        observations = observe(generate_case(config, 11), config, 101)
        for kind in ("classical", "quantum"):
            model = DarcyPINN(ModelConfig(features=2, hidden=6, feature_layers=1), kind, 101)
            errors = components(model, observations.training)
            total, _ = objective(errors, LossWeights(), torch.ones(config.collocation))
            self.assertEqual(errors["pressure"].numel(), 0)
            total.backward()
            for module in (model.permeability, model.pressure):
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_validation_uses_only_permeability_when_pressure_is_unobserved(self):
        config = small_data()
        observations = observe(generate_case(config, 11), config, 101)
        model = DarcyPINN(ModelConfig(features=2, hidden=6, feature_layers=1), "classical", 101)
        first = validation_score(model, observations.validation, LossWeights())
        with torch.no_grad():
            for parameter in model.pressure.parameters():
                parameter.fill_(1e5)
        self.assertEqual(first, validation_score(model, observations.validation, LossWeights()))

    def test_mc_rejects_a_mask_that_worsens_validation(self):
        data = small_data()
        observations = observe(generate_case(data, 11), data, 101)
        model = DarcyPINN(ModelConfig(features=4, hidden=6, feature_layers=1), "classical", 101)
        xy = observations.validation.k_xy
        validation = ValidationView(
            xy, model.permeability(tensor(xy)).detach().numpy(),
            xy, model.pressure(tensor(xy)).detach().numpy(),
        )
        original = model.permeability.feature_mask.clone()
        strategy = StrategyConfig(mc_trials=2, mc_batch=4, min_features=3, mc_threshold=1e6)
        event = screen_features(model, observations.training, validation, LossWeights(),
                                torch.ones(data.collocation, dtype=torch.float64), strategy,
                                np.random.default_rng(19))
        self.assertTrue(event["candidate_features"])
        self.assertGreater(event["validation_after"], event["validation_before"])
        self.assertFalse(event["accepted"])
        torch.testing.assert_close(original, model.permeability.feature_mask)

    def test_strategies_do_not_change_training_during_warmup(self):
        config = small_data()
        observations = observe(generate_case(config, 11), config, 101)
        common = (101, ModelConfig(features=2, hidden=6, feature_layers=1),
                  TrainConfig(epochs=2), StrategyConfig(warmup=10), LossWeights())
        plain = fit(observations.training, observations.validation, "classical_plain", *common)
        enhanced = fit(observations.training, observations.validation, "classical_rba_mc", *common)
        self.assertEqual(plain[2], enhanced[2])
        self.assertEqual(enhanced[3], [])
        for a, b in zip(plain[0].state_dict().values(), enhanced[0].state_dict().values()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_training_reproducibility_and_best_checkpoint(self):
        config = small_data()
        observations = observe(generate_case(config, 11), config, 101)
        args = (observations.training, observations.validation, "quantum_rba_mc", 101,
                ModelConfig(features=2, hidden=6, feature_layers=1),
                TrainConfig(epochs=3), StrategyConfig(warmup=0, mc_every=2, min_features=1,
                                                     mc_trials=2, mc_batch=4), LossWeights())
        first = fit(*args)
        second = fit(*args)
        for a, b in zip(first[0].state_dict().values(), second[0].state_dict().values()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[1]["best_validation"], min(row["validation"] for row in first[2]))


class ArchiveTests(unittest.TestCase):
    def test_full_factorial_archive_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run_experiment(
                output, small_data(), ModelConfig(features=2, feature_layers=1, hidden=6),
                TrainConfig(epochs=2), StrategyConfig(warmup=0, mc_every=1, mc_trials=2,
                                                     mc_batch=4, min_features=1),
                LossWeights(), [11], [101], list(METHODS), "unit-test",
            )
            checked = verify_experiment(output)
            self.assertEqual(checked["verified_runs"], 8)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["data_config"]["field_generator"], "gstools")
            self.assertEqual(manifest["loss_weights"]["pressure"], 0)
            self.assertIn("gstools", manifest["versions"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertIn("quantum_rba_mc_vs_classical_rba_mc", summary["paired_comparisons"])
            self.assertIsNone(summary["methods"]["quantum_plain"]["k_relative_l2"]["sd"])
            with self.assertRaises(FileExistsError):
                run_experiment(output, small_data(), ModelConfig(), TrainConfig(),
                               StrategyConfig(), LossWeights(), [11], [101], list(METHODS))
            reference = output / "field_11/reference.json"
            reference.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_experiment(output)

    def test_explicit_legacy_pressure_archive_still_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "legacy"
            config = replace(small_data(), field_generator="trigonometric",
                             p_train=8, p_validation=4)
            run_experiment(output, config, ModelConfig(features=2, hidden=6, feature_layers=1),
                           TrainConfig(epochs=1), StrategyConfig(min_features=1),
                           LossWeights(pressure=10), [11], [101], ["classical_plain"], "unit-test")
            self.assertEqual(verify_experiment(output)["verified_runs"], 1)

    def test_pressure_weight_must_match_observation_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "pressure observation"):
                run_experiment(Path(directory) / "invalid", small_data(), ModelConfig(),
                               TrainConfig(), StrategyConfig(), LossWeights(pressure=10),
                               [11], [101], ["classical_plain"], "unit-test")


if __name__ == "__main__":
    unittest.main()
