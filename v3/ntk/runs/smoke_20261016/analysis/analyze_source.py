"""只读核验原始归档并生成场级配对统计，不按NTK或测试指标重新选模。"""

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr
import torch

from core import GROUPS, analyze_checkpoint, kernel_statistics
from dataset.cases import DataConfig, generate_case, observe
from experiment.runner import digest, write_json
from metrics.physics import LossWeights
from models.baseline_model import BaselineDarcyPINN
from models.darcy_pinn import ModelConfig
from models.qmodel import QDarcyPINN
from qpinn.training import reconstruction_metrics


def write_csv(path, rows):
    """保持完整数值精度，所有图和报告均从同一组记录生成。"""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_summary(classical, quantum):
    """重复先在场内平均；bootstrap和精确符号翻转均以场为单位。"""
    classical, quantum = np.asarray(classical), np.asarray(quantum)
    differences = quantum - classical
    rng = np.random.default_rng(20261016)
    samples = rng.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
    ci = np.quantile(samples, [0.025, 0.975])
    if len(differences) > 16:
        raise ValueError("exact field-level sign-flip is limited to 16 fields")
    null = np.array([np.mean(differences * signs)
                     for signs in product((-1, 1), repeat=len(differences))])
    pvalue = float(np.mean(np.abs(null) >= abs(differences.mean()) - 1e-14))
    return {
        "independent_fields": len(differences),
        "classical_mean": float(classical.mean()), "quantum_mean": float(quantum.mean()),
        "mean_quantum_minus_classical": float(differences.mean()),
        "relative_reduction_of_means_percent": float(100 * (1 - quantum.mean() / classical.mean())),
        "field_bootstrap_95_percentile_ci_difference": ci.tolist(),
        "exact_field_sign_flip_two_sided_p": pvalue,
        "quantum_better_fields": int(np.sum(differences < 0)),
        "quantum_worse_fields": int(np.sum(differences > 0)),
        "field_differences": differences.tolist(),
    }


def verify_archive(output, manifest, records, deep=False):
    """复算雅可比乘积、谱、分组及预测指标；可选从权重重新自动微分。"""
    for name, expected in manifest["source_sha256"].items():
        if digest(output / "source" / name) != expected:
            raise ValueError(f"source hash mismatch: {name}")
    for name, expected in manifest["artifact_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    keys = [(r["field_seed"], r["seed"], r["method"], r["epoch"]) for r in records]
    if len(set(keys)) != len(keys) or len(keys) != manifest["snapshots"]:
        raise ValueError("duplicate or missing snapshot records")
    for field, seed, method in product(manifest["field_seeds"], manifest["seeds"], manifest["methods"]):
        rows = [r for r in records if (r["field_seed"], r["seed"], r["method"]) == (field, seed, method)]
        directory = output / f"field_{field}/seed_{seed}/{method}"
        history = json.loads((directory / "history.json").read_text())
        info = json.loads((directory / "training.json").read_text())
        best_epoch = min(history, key=lambda r: r["validation"])["epoch"]
        selected = [r for r in rows if r["is_best"]]
        if len(selected) != 1 or selected[0]["epoch"] != best_epoch or info["best_epoch"] != best_epoch:
            raise ValueError("checkpoint selection is not validation-only")
        expected_epochs = set(manifest["checkpoint_epochs"]) | {best_epoch}
        if {r["epoch"] for r in rows} != expected_epochs:
            raise ValueError("checkpoint schedule mismatch")
        initial = np.load(directory / "epoch_00000.npz", allow_pickle=False)
        with np.load(directory.parent / "observations.npz", allow_pickle=False) as obs, np.load(
                output / f"field_{field}/reference.npz", allow_pickle=False) as ref:
            case = SimpleNamespace(log_k=ref["log_k"], pressure=ref["pressure"])
            observations = SimpleNamespace(k_test_indices=obs["k_test_indices"], p_test_indices=obs["p_test_indices"])
            for row in rows:
                with np.load(directory / f"epoch_{row['epoch']:05d}.npz", allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays["probe_xy"], ref["xy"][ref["probe_indices"]])
                    np.testing.assert_array_equal(arrays["probe_truth"], ref["log_k"][ref["probe_indices"]])
                    for kind in ("output", "loss"):
                        jacobian, kernel = arrays[f"{kind}_jacobian"], arrays[f"{kind}_kernel"]
                        np.testing.assert_allclose(kernel, jacobian @ jacobian.T, atol=1e-11, rtol=1e-11)
                        np.testing.assert_allclose(kernel, sum(arrays[f"{kind}_kernel_{g}"] for g in GROUPS),
                                                   atol=1e-11, rtol=1e-11)
                        direction = (arrays["probe_truth"] - arrays["probe_truth"].mean()
                                     if kind == "output" else arrays["weighted_training_errors"])
                        statistics, spectrum = kernel_statistics(kernel, initial[f"{kind}_kernel"], direction)
                        np.testing.assert_allclose(spectrum, arrays[f"{kind}_eigenvalues"], atol=1e-11, rtol=1e-9)
                        for name, value in statistics.items():
                            if value is not None:
                                np.testing.assert_allclose(value, row["ntk"][kind][name], atol=1e-10, rtol=1e-9)
                    metrics = reconstruction_metrics(case, observations, arrays["prediction_log_k"],
                                                     arrays["prediction_pressure"])
                    for name, value in metrics.items():
                        np.testing.assert_allclose(value, row["metrics"][name], atol=1e-11, rtol=1e-10)
                    validation = manifest["loss_weights"]["permeability"] * np.mean(
                        (arrays["prediction_log_k"][obs["k_validation_indices"]] - obs["validation_log_k"])**2)
                    np.testing.assert_allclose(validation, row["validation"], atol=1e-12, rtol=1e-10)
                    for name, array_name in (("pde_residual_rmse", "prediction_physics_residual"),
                                             ("neumann_flux_rmse", "prediction_neumann_flux")):
                        np.testing.assert_allclose(np.sqrt(np.mean(arrays[array_name]**2)),
                                                   row["metrics"][name], atol=1e-12, rtol=1e-10)
        initial.close()
    deep_count = 0
    if deep:
        # 初场首次重复的两种架构，分别重载初始与最佳权重；不是全矩阵重训。
        torch.set_num_threads(1)
        field, seed = manifest["field_seeds"][0], manifest["seeds"][0]
        data = DataConfig(**manifest["data_config"])
        case = generate_case(data, field)
        obs = observe(case, data, seed)
        with np.load(output / f"field_{field}/reference.npz", allow_pickle=False) as ref:
            indices = ref["probe_indices"]
            np.testing.assert_array_equal(case.log_k, ref["log_k"])
        for method in manifest["methods"]:
            directory = output / f"field_{field}/seed_{seed}/{method}"
            bundle = torch.load(directory / "checkpoints.pt", weights_only=True, map_location="cpu")
            cls = QDarcyPINN if method.startswith("quantum") else BaselineDarcyPINN
            model = cls(ModelConfig(**manifest["model_config"]), seed)
            for epoch in sorted({0, bundle["best_epoch"]}):
                model.load_state_dict(bundle["states"][epoch])
                fresh = analyze_checkpoint(model, obs.training, LossWeights(**manifest["loss_weights"]),
                                           case.xy[indices], case.log_k[indices])[0]
                with np.load(directory / f"epoch_{epoch:05d}.npz", allow_pickle=False) as saved:
                    for name in ("output_jacobian", "loss_jacobian", "output_kernel", "loss_kernel"):
                        np.testing.assert_allclose(fresh[name], saved[name], atol=1e-11, rtol=1e-10)
                deep_count += 1
    return {"status": "passed", "verified_snapshots": len(records),
            "artifact_hashes": len(manifest["artifact_sha256"]),
            "source_hashes": len(manifest["source_sha256"]),
            "reloaded_autodiff_snapshots": deep_count}


def analyze(output, deep=False):
    """输出完整CSV和中文报告；量子未获胜时自动生成对应的否定或有限结论。"""
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("only completed archives can be analyzed")
    records = json.loads((output / "records.json").read_text())
    verification = verify_archive(output, manifest, records, deep)
    destination = output / "analysis"
    destination.mkdir(exist_ok=True)
    flat, frequencies, spectra, blocks, best = [], [], [], [], []
    for record in records:
        identity = {name: record[name] for name in ("field_seed", "seed", "method", "epoch", "is_best", "is_scheduled")}
        row = {**identity, "validation": record["validation"], **record["metrics"]}
        for kind, statistics in record["ntk"].items():
            row.update({f"{kind}_{name}": value for name, value in statistics.items()})
        flat.append(row)
        frequencies.extend({**identity, **item} for item in record["frequency"])
        directory = output / f"field_{record['field_seed']}/seed_{record['seed']}/{record['method']}"
        with np.load(directory / f"epoch_{record['epoch']:05d}.npz", allow_pickle=False) as arrays:
            for kind in ("output", "loss"):
                eigenvalues = arrays[f"{kind}_eigenvalues"]
                spectra.extend({**identity, "kernel": kind, "rank": i + 1, "eigenvalue": float(value),
                                "trace_normalized": float(value / record["ntk"][kind]["trace"])}
                               for i, value in enumerate(eigenvalues))
            # 保留数据、PDE及边界之间的交叉核，辅助识别联合训练的任务尺度失衡。
            kernel = arrays["loss_kernel"]
            slices = record["metadata"]["loss_slices"]
            for left, (a, b) in slices.items():
                for right, (c, d) in slices.items():
                    block = kernel[a:b, c:d]
                    blocks.append({
                        **identity, "row_task": left, "column_task": right,
                        "rows": b - a, "columns": d - c,
                        "frobenius_norm": float(np.linalg.norm(block)),
                        "frobenius_fraction_of_full_kernel": float(np.linalg.norm(block) / np.linalg.norm(kernel)),
                        "diagonal_trace_fraction": (float(np.trace(block) / np.trace(kernel))
                                                    if left == right else None),
                    })
        if record["is_best"]:
            info = json.loads((directory / "training.json").read_text())
            best.append({**row, "training_seconds": info["training_seconds"],
                         "diagnostic_seconds": info["diagnostic_seconds"],
                         "total_parameters": info["parameter_counts"]["total"],
                         "feature_parameters": info["parameter_counts"]["feature_block"]})
    write_csv(destination / "checkpoint_metrics.csv", flat)
    write_csv(destination / "frequency_responses.csv", frequencies)
    write_csv(destination / "eigenspectra.csv", spectra)
    write_csv(destination / "loss_kernel_blocks.csv", blocks)
    write_csv(destination / "best_runs.csv", best)
    fields = []
    for field in manifest["field_seeds"]:
        row = {"field_seed": field, "repeats": len(manifest["seeds"])}
        for name in ("k_relative_l2", "log_k_rmse", "pressure_rmse", "pde_residual_rmse",
                     "downstream_flux_relative_error", "training_seconds", "output_raw_drift",
                     "output_shape_drift", "output_direction_alignment"):
            for method in manifest["methods"]:
                values = [r[name] for r in best if r["field_seed"] == field and r["method"] == method]
                row[f"{method}_{name}"] = float(np.mean(values))
            row[f"quantum_minus_classical_{name}"] = row[f"quantum_plain_{name}"] - row[f"classical_plain_{name}"]
        fields.append(row)
    write_csv(destination / "field_paired.csv", fields)
    main = paired_summary([r["classical_plain_k_relative_l2"] for r in fields],
                          [r["quantum_plain_k_relative_l2"] for r in fields])
    lookup = {(r["field_seed"], r["seed"], r["method"]): r for r in best}
    wins = sum(lookup[f, s, "quantum_plain"]["k_relative_l2"] < lookup[f, s, "classical_plain"]["k_relative_l2"]
               for f, s in product(manifest["field_seeds"], manifest["seeds"]))
    associations = {}
    for method in manifest["methods"]:
        for metric in ("output_raw_drift", "output_shape_drift", "output_direction_alignment"):
            x = [r[f"{method}_{metric}"] for r in fields]
            y = [r[f"{method}_k_relative_l2"] for r in fields]
            rho = float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else None
            associations[f"{method}:{metric}"] = rho
    summary = {"profile": manifest["profile"], "primary": main, "paired_run_wins": int(wins),
               "paired_runs": len(best) // 2, "exploratory_field_spearman": associations,
               "best_at_budget_end": {method: sum(r["method"] == method and r["epoch"] == manifest["train_config"]["epochs"]
                                                   for r in best) for method in manifest["methods"]},
               "secondary_means": {method: {name: float(np.mean([r[name] for r in best if r["method"] == method]))
                                           for name in ("log_k_rmse", "pressure_rmse", "pde_residual_rmse",
                                                        "downstream_flux_relative_error", "training_seconds",
                                                        "output_raw_drift", "output_shape_drift",
                                                        "output_direction_alignment")}
                                   for method in manifest["methods"]},
               "verification": verification, "analysis_source_sha256": digest(Path(__file__))}
    write_json(destination / "summary.json", summary)
    write_json(destination / "verification.json", verification)
    lo, hi = main["field_bootstrap_95_percentile_ci_difference"]
    if main["mean_quantum_minus_classical"] < 0:
        conclusion = "量子组在本轮固定协议下平均留出误差较低。"
    else:
        conclusion = "本轮固定协议未显示量子组平均留出误差优势。"
    if lo <= 0 <= hi:
        conclusion += "场级bootstrap区间跨零，不能排除收益不稳定或无收益。"
    if manifest["profile"] == "smoke":
        conclusion = "本归档仅检查流程与数值正确性，不能用于架构优越性结论。"
    lines = [
        "# V3 有限宽度经验 NTK 实验报告", "",
        f"## 结论\n\n{conclusion}",
        "本结果不证明量子模块普遍更优，不证明纠缠必要性、效率优势或量子计算优势。", "",
        "## 固定实验",
        f"- 场种子：{manifest['field_seeds']}；每场重复：{manifest['seeds']}。",
        f"- {manifest['expected_runs']}次训练；每次{manifest['train_config']['epochs']}轮；关闭RBA/MC。",
        f"- {len(records)}个检查点；原始J、K、分组核、谱、误差与权重完整保存。",
        "- 验证集选模；测试真值只进入离线诊断，不改变训练或选择。",
        "- CPU双精度精确量子模拟；相同轮数但不是等参数或等墙钟成本。",
        "", "## 主要终点",
        "| 指标 | 数值 |", "| --- | ---: |",
        f"| 经典平均测试渗透率相对L2 | {main['classical_mean']:.8f} |",
        f"| 量子平均测试渗透率相对L2 | {main['quantum_mean']:.8f} |",
        f"| 均值相对降低百分比 | {main['relative_reduction_of_means_percent']:.4f}% |",
        f"| 配对差值Q-C | {main['mean_quantum_minus_classical']:.8f} |",
        f"| 场级bootstrap 95%百分位区间 | [{lo:.8f}, {hi:.8f}] |",
        f"| 场级双侧精确符号翻转p | {main['exact_field_sign_flip_two_sided_p']:.6f} |",
        f"| 量子较好配对重复 | {wins}/{len(best)//2} |",
        f"| 量子较好独立场（场内平均） | {main['quantum_better_fields']}/{len(fields)} |",
        "", "## 各场配对结果",
        "| 场 | 经典 | 量子 | Q-C |", "| --- | ---: | ---: | ---: |",
    ]
    for row in fields:
        lines.append(f"| {row['field_seed']} | {row['classical_plain_k_relative_l2']:.6f} | "
                     f"{row['quantum_plain_k_relative_l2']:.6f} | {row['quantum_minus_classical_k_relative_l2']:.6f} |")
    lines += ["", "## NTK与训练代价",
              "| 架构 | 原始输出核变化 | 归一化结构变化 | 目标方向对齐 | 训练秒 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for method, values in summary["secondary_means"].items():
        lines.append(f"| {method} | {values['output_raw_drift']:.5f} | {values['output_shape_drift']:.5f} | "
                     f"{values['output_direction_alignment']:.5f} | {values['training_seconds']:.4f} |")
    lines += ["", "## 次要终点（描述性均值）",
              "| 架构 | log_k RMSE | 压力RMSE | PDE残差RMSE | 下游通量相对误差 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for method, values in summary["secondary_means"].items():
        lines.append(f"| {method} | {values['log_k_rmse']:.6f} | {values['pressure_rmse']:.6f} | "
                     f"{values['pde_residual_rmse']:.6f} | {values['downstream_flux_relative_error']:.6f} |")
    lines += ["", "原始变化同时受核尺度和结构影响；变化更大不等于性能更好。",
              "目标方向对齐为v^TKv/(trK||v||²)，v为探针位置去均值log_k，只作事后机制诊断。",
              "联合损失方向使用当前训练误差，模型之间方向不完全相同，不据此单独判断架构。",
              "NTK与误差的场级Spearman相关保存在summary.json，仅为小样本探索性描述。",
              "", "## 数值复核",
              f"- 原始产物哈希：{verification['artifact_hashes']}；源码哈希：{verification['source_hashes']}。",
              f"- 复核{verification['verified_snapshots']}份核的J乘积、谱、分组加和及预测指标。",
              f"- 从权重独立重新自动微分：{verification['reloaded_autodiff_snapshots']}个检查点，不是全矩阵重训。",
              "", "## 文件",
              "- `best_runs.csv`：验证最优模型的逐次结果。",
              "- `field_paired.csv`：以独立场为单位的配对结果。",
              "- `checkpoint_metrics.csv`：全部检查点的误差、核尺度、结构变化和分组贡献。",
              "- `frequency_responses.csv`、`eigenspectra.csv`：频率响应与完整谱。",
              "- `loss_kernel_blocks.csv`：观测、PDE和边界任务的对角及交叉核块。",
              "- `summary.json`、`verification.json`：统计与复核。",
              "", "## 解释限制",
              "6个场仍是小样本，bootstrap区间仅为探索性估计；符号翻转检验依赖场级差异的对称/交换性假设。",
              "次要终点及NTK关联没有多重比较校正，不据此宣称显著。",
              "需补等成本、强周期基线及无纠缠消融，才能进一步定位优势来源。",
              f"验证最优轮次落在预算末端的次数：{summary['best_at_budget_end']}；不预设已充分收敛。",
              "每个场均重新训练，不能描述为无需适配的新场泛化。",
              "有限宽度、Adam和梯度裁剪下的经验核不是固定核收敛证明。"]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 分析脚本独立冻结，报告排版改动不回写训练源码快照和原始产物哈希。
    analysis_source = destination / "analyze_source.py"
    if analysis_source.resolve() != Path(__file__).resolve():
        shutil.copyfile(Path(__file__), analysis_source)
    print(json.dumps(summary["primary"], indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    analyze(args.output, args.deep)
