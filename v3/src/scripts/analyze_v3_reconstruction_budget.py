"""Compare matched training budgets without changing either frozen archive."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import analyze_v3_gstools_results as plots


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "v3/experiments/quantum_pinn_v3_gstools_konly_20261007"
NEW = ROOT / "v3/experiments/quantum_pinn_v3_reconstruction_budget_20261007"
OUT = ROOT / "v3/paper/figures/manuscript_v3_budget5000"
METHODS = ("classical_plain", "quantum_plain")
METRICS = ("k_relative_l2", "log_k_rmse", "pde_residual_rmse",
           "best_validation", "best_epoch", "training_seconds")


def checked_manifest(folder):
    manifest = plots.read(folder / "manifest.json")
    assert manifest["status"] == "completed", folder
    for prefix, key in (("source", "source_sha256"), ("", "artifact_sha256")):
        for relative, expected in manifest[key].items():
            path = (folder / prefix / relative).resolve()
            assert path.is_relative_to(folder)
            assert plots.digest(path) == expected, path
    return manifest


def evidence():
    old, new = checked_manifest(OLD), checked_manifest(NEW)
    for key in ("data_config", "model_config", "loss_weights", "strategy_config",
                "source_sha256", "field_seeds", "versions"):
        assert old[key] == new[key], key
    assert new["seeds"] == [old["seeds"][0]]
    assert new["methods"] == list(METHODS)
    assert new["train_config"]["epochs"] == 5000
    assert old["train_config"]["epochs"] == 300
    assert all(new["train_config"][k] == v for k, v in old["train_config"].items()
               if k != "epochs")
    seed = new["seeds"][0]
    rows, comparison, data = [], [], {}
    for field in new["field_seeds"]:
        old_field, new_field = OLD / f"field_{field}", NEW / f"field_{field}"
        with np.load(old_field / "reference.npz") as a, np.load(new_field / "reference.npz") as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            xy, truth = b["xy"], np.exp(b["log_k"])
        relative = f"seed_{seed}/observations.npz"
        with np.load(old_field / relative) as a, np.load(new_field / relative) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            splits = np.full(len(truth), "test", dtype="<U10")
            splits[b["k_train_indices"]] = "train"
            splits[b["k_validation_indices"]] = "validation"
        item = {"xy": xy, "truth": truth, "splits": splits}
        for method in METHODS:
            relative = f"seed_{seed}/{method}"
            old_run, new_run = old_field / relative, new_field / relative
            a, b = plots.read(old_run / "result.json"), plots.read(new_run / "result.json")
            old_history = plots.read(old_run / "history.json")
            new_history = plots.read(new_run / "history.json")
            assert new_history[:len(old_history)] == old_history, (field, method, "prefix")
            best = min(new_history, key=lambda entry: entry["validation"])
            assert len(new_history) == new["train_config"]["epochs"] + 1
            assert b["best_epoch"] == best["epoch"]
            assert b["best_validation"] == best["validation"]
            row = {"field": field, "seed": seed, "method": method}
            for metric in METRICS:
                row[f"old_{metric}"], row[f"new_{metric}"] = a[metric], b[metric]
            row["relative_k_error_reduction"] = 1 - b["k_relative_l2"] / a["k_relative_l2"]
            comparison.append(row)
            with np.load(new_run / "predictions.npz") as pred:
                item[method] = np.exp(pred["log_k"])
            test = splits == "test"
            measured = np.linalg.norm(item[method][test] - truth[test]) / np.linalg.norm(truth[test])
            np.testing.assert_allclose(measured, b["k_relative_l2"], rtol=1e-13, atol=1e-14)
        data[field] = item
        for i in range(len(truth)):
            rows.append({
                "field": field, "seed": seed, "index": i, "x": xy[i, 0], "y": xy[i, 1],
                "split": splits[i], "reference_k": truth[i],
                **{method: item[method][i] for method in METHODS},
            })
    return new, data, rows, comparison


def audit():
    name = "fig04_spatial_maps"
    pdf = OUT / f"{name}.pdf"

    def run(script, args):
        result = subprocess.run([sys.executable, str(plots.AUDIT_SCRIPTS / script), *map(str, args)],
                                check=True, text=True, capture_output=True)
        return result.stdout

    preflight = json.loads(run("validate_figure.py", [Path(plots.__file__), "--json"]))
    plots.write_json(OUT / "source_preflight.json", preflight)
    text = json.loads(run("audit_pdf_text.py", [pdf, "--min-pt", "5", "--json"]))
    plots.write_json(OUT / f"{name}.text-audit.json", text)
    run("audit_figure_collisions.py", [
        pdf, "--json-out", OUT / f"{name}.collision-audit.json",
    ])
    collision = plots.read(OUT / f"{name}.collision-audit.json")
    assert collision["verdict"] == "PASS"
    return {"pdf_sha256": plots.digest(pdf), "collision_verdict": collision["verdict"],
            "text_exit_code": 0, "source_preflight": preflight["summary"]}


def write_report(comparison):
    lines = [
        "# V3 重建预算对比", "",
        "这是同一数据上的探索性预算诊断，不是独立外部验证。",
        "只改变预算300→5000轮；全部三个场、首次重复101、经典和量子基础组均保留。",
        "每次训练仅按验证损失选择检查点，测试值不进入训练或选模。",
        "参考场、全部观测及配置点逐数组一致，六次训练的前301条日志与旧归档逐记录一致。",
        "", "| 场 | 方法 | 原测试相对L2 | 新测试相对L2 | 误差相对下降 | 新选中轮次 | 原PDE RMSE | 新PDE RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['field']} | {row['method']} | {row['old_k_relative_l2']:.6f} | "
            f"{row['new_k_relative_l2']:.6f} | {row['relative_k_error_reduction']:.2%} | "
            f"{row['new_best_epoch']} | {row['old_pde_residual_rmse']:.6f} | "
            f"{row['new_pde_residual_rmse']:.6f} |"
        )
    lines += ["", "## 同次重复的跨场均值", ""]
    for method in METHODS:
        subset = [row for row in comparison if row["method"] == method]
        before = np.mean([row["old_k_relative_l2"] for row in subset])
        after = np.mean([row["new_k_relative_l2"] for row in subset])
        lines.append(f"- {method}：{before:.6f}→{after:.6f}，误差相对下降{1-after/before:.2%}。")
    lines += [
        "", "以上均值各含三个场的一次拟合，不是原论文每组九次拟合的均值。",
        "不能把延长训练的收益全部归因于量子模块，也不能据此宣布收敛或量子优势。",
        "本轮没有运行长预算RBA、MC或组合消融。",
        "", "## 新预算原始预测图", "",
        "![延长预算的原场、C、Q和测试误差](fig04_spatial_maps.png)",
        "",
        "每列为一个场；五行依次为原场、C、Q、C测试误差、Q测试误差。",
        "前三行共用渗透率色标，后两行共用误差色标，灰色为训练或验证位置。",
        "不裁剪、不插值、不独立归一化预测图、不混合参考场。",
        "",
        "## 核验范围", "",
        "图源共1728行，记录全部位置、划分和两个模型输出。独立指标复算及档案核验另由qpinn verify完成。",
        "面板对齐、最终PDF文字和碰撞检查已执行，详见同目录JSON报告。",
        "源代码预检中的未导出TIFF及正值断言识别警告沿用主绘图脚本的说明。",
        "本图复用既有Python/Matplotlib空间图布局；PNG为600 dpi，另有PDF/SVG。",
        "不能将自动检查视为期刊合规认证；最终语义和空间形态仍须目视检查。",
    ]
    (OUT / "RECONSTRUCTION_BUDGET_REPORT.md").write_text("\n".join(lines) + "\n")


def verify_saved(rows, comparison):
    metadata = plots.read(OUT / "budget_figures.json")
    for folder in (OLD, NEW):
        assert metadata["archive_manifest_sha256"][folder.name] == plots.digest(folder / "manifest.json")
    # 原绘图脚本已迁移；原哈希仍在元数据中，以下只复核归档和图源数值。
    assert metadata["source_sha256"]
    for relative, expected in metadata["artifact_sha256"].items():
        path = (OUT / relative).resolve()
        assert path.is_relative_to(OUT)
        assert plots.digest(path) == expected
    for name, expected in (("spatial_results_source.csv", rows), ("budget_comparison.csv", comparison)):
        with (OUT / name).open(newline="") as stream:
            actual = list(csv.DictReader(stream))
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert set(a) == set(b)
            for key, value in b.items():
                if isinstance(value, str):
                    assert a[key] == value
                else:
                    np.testing.assert_allclose(float(a[key]), value, rtol=1e-13, atol=1e-14)
    assert metadata["audit"]["pdf_sha256"] == plots.digest(OUT / "fig04_spatial_maps.pdf")
    return {"status": "passed", "source_check": "historical hashes retained; relocated sources not byte-identical",
            "runs": len(comparison), "spatial_rows": len(rows),
            "prefix_epochs_checked": 300, "archives": list(metadata["archive_manifest_sha256"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest, data, rows, comparison = evidence()
    if not args.verify_only:
        OUT.mkdir(parents=True, exist_ok=True)
        plots.OUT = OUT
        plots.FIGURE_RECORDS.clear()
        plots.setup_font()
        plots.write_csv("spatial_results_source.csv", rows)
        plots.write_csv("budget_comparison.csv", comparison)
        plots.spatial_maps(manifest, data)
        audit_result = audit()
        write_report(comparison)
        metadata = {
            "archive_manifest_sha256": {
                folder.name: plots.digest(folder / "manifest.json") for folder in (OLD, NEW)
            },
            "source_sha256": {
                str(path.relative_to(ROOT)): plots.digest(path)
                for path in (Path(__file__).resolve(), Path(plots.__file__).resolve())
            },
            "figures": plots.FIGURE_RECORDS, "audit": audit_result,
            "scope": "exploratory budget-only comparison on previously inspected fields; six fits",
            "artifact_sha256": {
                path.name: plots.digest(path) for path in sorted(OUT.iterdir())
                if path.is_file() and path.name != "budget_figures.json"
            },
        }
        plots.write_json(OUT / "budget_figures.json", metadata)
    print(json.dumps(verify_saved(rows, comparison), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
