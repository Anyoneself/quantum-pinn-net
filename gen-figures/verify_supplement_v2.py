from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/supplement_v2"
FIG = ROOT / "figures/manuscript_v2"


def check_tables(text):
    aggregate = json.loads((FIG / "summary.json").read_text())
    lookup = {(r["scenario"], r["method"], r["stage"]): r for r in aggregate}
    comparisons = [("fourier", "raw"), ("phase", "raw"), ("fourier", "corrected"),
                   ("phase", "corrected"), ("idw", "raw"), ("rbf", "raw")]
    ablations = ["phase", "no-mixer", "no-phase", "no-sine", "no-gate",
                 "no-condition", "data-only", "rba", "dynamic", "full"]
    specifications = {
        3: [[("random70", m, stage, key, 1, 5)
             for key in ("rmse", "mae", "r2", "high_gradient_rmse")]
            for m, stage in comparisons],
        4: [[("random70", m, stage, "rmse", 1, 5) for stage in ("raw", "corrected")]
            for m in ablations],
        5: [[(scenario, m, stage, "rmse", 1, 5) for m, stage in comparisons[2:]]
            for scenario in ("random70", "noise05", "noise10", "block")],
        6: [[(scenario, m, stage, key, scale, decimals)
             for scenario in ("random70", "block")
             for key, scale, decimals in (("pressure_rmse_mean", 1000, 3),
                                          ("flux_relative_error_mean", 100, 5))]
            for m, stage in comparisons],
    }
    checked, errors = 0, []
    for number, expected_rows in specifications.items():
        block = re.search(rf"(?m)^\*\*表{number} .*?\n\n((?:\|.*\n)+)", text).group(1)
        rows = [line.strip("|").split("|")[1:] for line in block.strip().splitlines()[2:]]
        assert len(rows) == len(expected_rows), number
        for row, expected in zip(rows, expected_rows):
            assert len(row) == len(expected), (number, row)
            for cell, (scenario, method, stage, key, scale, decimals) in zip(row, expected):
                item = lookup[scenario, method, stage]
                correct = (f"{item[key + '_mean'] * scale:.{decimals}f}"
                           f"±{item[key + '_sd'] * scale:.{decimals}f}")
                if cell.strip() != correct:
                    errors.append((number, cell.strip(), correct))
                checked += 1
    assert not errors, errors
    return checked


def check_curve_sources(truth):
    seeds = [101, 202, 303]
    comparisons = [("fourier", "raw"), ("phase", "raw"), ("fourier", "corrected"),
                   ("phase", "corrected"), ("idw", "raw"), ("rbf", "raw")]
    checked = 0

    def check_band(item, expected):
        nonlocal checked
        expected = np.asarray(expected)
        assert np.all(np.diff(item["x"]) > 0)
        np.testing.assert_allclose(item["runs"], expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(item["mean"], expected.mean(0), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(item["sd"], expected.std(0, ddof=1), rtol=1e-12, atol=1e-14)
        checked += 1

    def history(method, scenario):
        return [[h["validation_mse"] ** 0.5
                 for h in json.loads((EXP / f"seed_{seed}/{scenario}/{method}/history.json").read_text())]
                for seed in seeds]

    overall = json.loads((FIG / "overall_curves.json").read_text())
    for item, (method, stage) in zip(overall, comparisons):
        expected = [sorted(json.loads((EXP / f"seed_{seed}/random70/{method}/result.json").read_text())
                           [stage]["per_field_rmse"]) for seed in seeds]
        check_band(item, expected)
        np.testing.assert_allclose(item["x"], np.arange(1, 129) / 128 * 100)
    assert len(overall) == 6

    for item in json.loads((FIG / "ablation_curves.json").read_text()):
        check_band(item, history(item["method"], "random70"))
        assert item["x"] == list(range(1, 41))

    profiles = json.loads((FIG / "spatial_profiles.json").read_text())
    mask = np.load(EXP / "seed_101/block/observations.npz")["test"]
    for item in profiles["series"]:
        field, method, stage = item["field"], item["method"], item["stage"]
        pred = np.load(EXP / f"seed_101/block/{method}/{stage}.npy")[field, 16, 9:23]
        actual = truth[field, 16, 9:23]
        assert mask[field, 16, 9:23].all()
        np.testing.assert_array_equal(item["prediction"], pred)
        np.testing.assert_array_equal(item["reference"], actual)
        # The displayed profile preserves the original float32 subtraction.
        np.testing.assert_array_equal(item["signed_error"], pred - actual.astype(pred.dtype))
        checked += 1
    assert len(profiles["series"]) == 12

    sensitivity = json.loads((FIG / "sensitivity_curves.json").read_text())
    for item, scenario in zip(sensitivity["noise_validation"], ["random70", "noise05", "noise10"]):
        check_band(item, history("phase", scenario))
    for item in sensitivity["gap_columns"]:
        expected = []
        for seed in seeds:
            pred = np.load(EXP / f"seed_{seed}/block/{item['method']}/{item['stage']}.npy")
            error = pred[:, 9:23, 9:23].astype(float) - truth[:, 9:23, 9:23]
            expected.append(np.sqrt(np.mean(error ** 2, axis=(0, 1))))
        check_band(item, expected)

    for item in json.loads((FIG / "budget_curves.json").read_text()):
        path = EXP / f"seed_101/{item['scenario']}/{item['method']}/history.json"
        rows = json.loads(path.read_text())
        np.testing.assert_array_equal(item["x"], [h["epoch"] for h in rows])
        np.testing.assert_array_equal(item["validation_rmse"], [h["validation_mse"] ** 0.5 for h in rows])
        checked += 1

    flow = json.loads((FIG / "flow_profiles.json").read_text())
    assert len(flow["series"]) == 24 and len(flow["metric_checks"]) == 36
    for item in flow["series"]:
        check_band(item, item["runs"])
        np.testing.assert_allclose(item["x"], (np.arange(32) + 0.5) / 32)
        assert np.asarray(item["runs"]).shape == (3, 32)
    for item in flow["metric_checks"]:
        assert item["max_pressure_metric_difference"] < 1e-12
        assert item["max_flux_metric_difference"] < 1e-12
    return checked


def main():
    manifest = json.loads((EXP / "manifest.json").read_text())
    source = ROOT / "experiments/rba_qpinn_submission_v1/seed_42/processed/fields.npy"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["data_sha256"]
    conditions = source.with_name("conditions.npy")
    assert hashlib.sha256(conditions.read_bytes()).hexdigest() == manifest["condition_sha256"]
    runner = ROOT / "gen-figures/run_supplement_v2.py"
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == manifest["script_sha256"]
    truth = np.load(source).astype(float)
    paths = sorted(EXP.glob("seed_*/*/*/result.json"))
    assert len(paths) == 101, len(paths)
    max_difference = 0.0
    neural = 0
    output_arrays = 0
    for path in paths:
        r = json.loads(path.read_text())
        scenario = "random70" if r["scenario"] == "budget120" else r["scenario"]
        observation = np.load(EXP / f"seed_{r['seed']}/{scenario}/observations.npz")
        train, val, test = [observation[k] for k in ("train", "validation", "test")]
        assert not np.any(train & val | train & test | val & test)
        assert np.all(observation["observed"][test] == 0)
        for stage in ("raw", "corrected"):
            if stage not in r:
                continue
            pred = np.load(path.parent / f"{stage}.npy").astype(float)
            assert pred.shape == truth.shape and np.isfinite(pred).all()
            output_arrays += 1
            error = (pred - truth)[test]
            actual = truth[test]
            checks = {
                "rmse": np.sqrt(np.mean(error ** 2)),
                "mae": np.mean(abs(error)),
                "r2": 1 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2),
            }
            for key, expected in checks.items():
                difference = abs(expected - r[stage][key])
                assert difference < 1e-6, (path, stage, key, difference)
                max_difference = max(max_difference, difference)
            per_field = np.array([np.sqrt(np.mean((pred[s][test[s]] - truth[s][test[s]]) ** 2))
                                  for s in range(len(truth))])
            np.testing.assert_allclose(per_field, r[stage]["per_field_rmse"], atol=1e-7)
            if stage == "corrected":
                np.testing.assert_array_equal(pred[train], observation["observed"][train])
            flow = r.get(f"flow_{stage}", {})
            if flow:
                assert flow["max_solver_residual"] < 1e-10
                assert flow["max_flux_imbalance"] < 1e-10
        if r["method"] not in ("idw", "rbf"):
            neural += 1
            history = json.loads((path.parent / "history.json").read_text())
            assert len(history) == r["epochs"]
            best = min(history, key=lambda h: h["validation_mse"])
            assert best["epoch"] == r["best_epoch"]
            pred = np.load(path.parent / "raw.npy")
            validation_mse = float(np.mean((pred[val] - observation["observed"][val]) ** 2))
            assert abs(validation_mse - r["validation_mse"]) < 1e-7
        else:
            search = json.loads((path.parent / "history.json").read_text())
            best = min(search["parameter_search"], key=lambda h: h["validation_mse"])
            assert best["parameters"] == search["selected"] == r["selected"]
    paper = ROOT / "paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md"
    text = paper.read_text()
    assert not re.search(r"<!-- .*AFTER_|TODO|待补结果", text)
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert len(figures) == 10
    assert "../figures/architecture/phase_periodic_network.png" in figures
    assert "../figures/architecture/training_workflow.png" in figures
    for image in figures:
        assert (paper.parent / image).resolve().is_file(), image
    tags = [int(n) for n in re.findall(r"\\tag\{(\d+)\}", text)]
    assert tags == list(range(1, 19)), tags
    body = text.split("## 0 引言", 1)[1].split("## 数据与代码说明", 1)[0]
    han = len(re.findall(r"[\u3400-\u9fff]", body))
    table_cells = check_tables(text)
    curve_series = check_curve_sources(truth)
    assert not re.search(r"观测率|观测比例|比例观测|观测密度|(?:20|40|70)%", text)
    reference_numbers = [int(x) for x in re.findall(r"(?m)^\[(\d+)\] ", text)]
    assert len(reference_numbers) >= 30
    assert reference_numbers == list(range(1, len(reference_numbers) + 1))
    summary = {"completed_runs": len(paths), "neural_training_runs": neural,
               "evaluated_output_arrays": output_arrays, "field_count": len(truth),
               "max_recomputed_metric_difference": max_difference, "main_text_han_characters": han,
               "main_figures": len(figures), "equations": len(tags),
               "result_table_cells_checked": table_cells, "references": len(reference_numbers),
               "curve_series_checked": curve_series,
               "seed_word_mentions_in_main_text": body.count("种子")}
    (FIG / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
