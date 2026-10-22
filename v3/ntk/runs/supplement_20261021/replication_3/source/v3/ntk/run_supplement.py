"""固定八组补充实验，复用配对训练与核验，不按测试表现筛选结果。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from analyze_results import analyze, paired_summary, write_csv
from dataset.cases import DataConfig
from experiment.runner import ROOT, digest, write_json
from models.darcy_pinn import ModelConfig
from run_experiment import run, source_snapshot


def groups(smoke=False):
    """场种子及单因素变化在训练前固定；条件组复用第一组场种子。"""
    specs = [
        (f"replication_{i}", [100 * (i + 4) + 11 * j for j in range(1, 7)], {})
        for i in range(1, 5)
    ]
    specs += [
        ("sparse", specs[0][1], {"k_train": 24}),
        ("noise10", specs[0][1], {"noise": 0.1}),
        ("short_correlation", specs[0][1], {"correlation_length": 0.125}),
        ("strong_anisotropy", specs[0][1], {"anisotropy": 0.3}),
    ]
    result = []
    for name, fields, changes in specs:
        data = DataConfig()
        if smoke:
            data = replace(data, grid=12, k_train=8, k_validation=4, collocation=4, boundary=2)
            if name == "sparse":
                changes = {"k_train": 4}
        data = replace(data, **changes)
        data.validate()
        result.append({
            "name": name, "field_seeds": fields[:1] if smoke else fields,
            "seeds": [101] if smoke else [101, 202, 303],
            "data_config": asdict(data),
        })
    return result


def worker(output, name, smoke):
    """独立进程顺序拟合一组，随后逐产物核验并重新自动微分抽查。"""
    spec = next(group for group in groups(smoke) if group["name"] == name)
    run(output, spec["field_seeds"], spec["seeds"],
        epochs=2 if smoke else 1000,
        checkpoints=(0, 1, 2) if smoke else (0, 100, 300, 1000),
        data_config=DataConfig(**spec["data_config"]),
        model_config=ModelConfig(features=2, hidden=6, feature_layers=1) if smoke else None,
        probe_grid=4 if smoke else 8, profile="smoke" if smoke else "supplement")
    analyze(output, deep=True)


def aggregate(output, plan):
    """只汇总预设且全部核验通过的组，主检验排除旧场与变化条件。"""
    descriptions, field_rows, summaries, artifact_hashes = [], [], {}, {}
    for spec in plan["groups"]:
        directory = output / spec["name"]
        manifest = json.loads((directory / "manifest.json").read_text())
        for key in ("field_seeds", "seeds", "data_config"):
            if manifest[key] != spec[key]:
                raise ValueError(f"group configuration mismatch: {spec['name']}:{key}")
        summary = json.loads((directory / "analysis/summary.json").read_text())
        if (manifest["status"] != "completed"
                or manifest["completed_runs"] != len(spec["field_seeds"]) * len(spec["seeds"]) * 2
                or summary["verification"]["status"] != "passed"
                or summary["verification"]["reloaded_autodiff_snapshots"] < 2):
            raise ValueError(f"group is incomplete or unverified: {spec['name']}")
        summaries[spec["name"]] = summary
        primary = summary["primary"]
        descriptions.append({"group": spec["name"], **{
            key: value for key, value in primary.items() if not isinstance(value, list)}})
        # 场级均值来自验证最优模型，所有场差值与源记录一起保留。
        with (directory / "analysis/field_paired.csv").open(newline="") as handle:
            field_rows.extend({"group": spec["name"], **row} for row in csv.DictReader(handle))
        for name in ("manifest.json", "records.json", "analysis/summary.json",
                     "analysis/field_paired.csv", "analysis/verification.json"):
            artifact_hashes[str((directory / name).relative_to(output))] = digest(directory / name)
    pooled = [row for row in field_rows if row["group"].startswith("replication_")]
    if len({row["field_seed"] for row in pooled}) != len(pooled):
        raise ValueError("replication fields must be independent and unique")
    primary = paired_summary(
        [float(row["classical_plain_k_relative_l2"]) for row in pooled],
        [float(row["quantum_plain_k_relative_l2"]) for row in pooled])
    tests = {"replication_pooled": primary}
    tests.update({name: summary["primary"] for name, summary in summaries.items()
                  if not name.startswith("replication_")})
    for comparison in tests.values():
        comparison["bonferroni_p_five_comparisons"] = min(
            1.0, len(tests) * comparison["exact_field_sign_flip_two_sided_p"])
    write_csv(output / "groups.csv", descriptions)
    write_csv(output / "fields.csv", field_rows)
    write_json(output / "summary.json", {
        "profile": plan["profile"], "primary": primary, "planned_comparisons": tests,
        "groups": summaries, "plan_sha256": digest(output / "plan.json"),
        "group_artifact_sha256": artifact_hashes,
        "completed_runs": plan["expected_runs"],
    })
    rows = [
        "# V3 八组补充实验报告", "",
        "## 主要复验", "",
        ("本归档是缩小配置流程检查，不用于科研结论。" if plan["profile"] == "smoke" else
         "四个同配置组的全部24个新场构成主要复验；旧六场单独保留，不进入本次统计。"),
        "",
        f"经典平均留出误差{primary['classical_mean']:.8f}，量子{primary['quantum_mean']:.8f}，"
        f"均值相对降低{primary['relative_reduction_of_means_percent']:.4f}%；"
        f"{primary['quantum_better_fields']}/{primary['independent_fields']}场的量子场内均值更低。",
        "", "## 五项预设比较", "",
        "| 比较 | 场数 | C | Q | 降低% | Q-C区间 | 精确p | Bonferroni p |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, values in tests.items():
        lo, hi = values["field_bootstrap_95_percentile_ci_difference"]
        rows.append(
            f"| {name} | {values['independent_fields']} | {values['classical_mean']:.6f} | "
            f"{values['quantum_mean']:.6f} | {values['relative_reduction_of_means_percent']:.2f} | "
            f"[{lo:.6f}, {hi:.6f}] | {values['exact_field_sign_flip_two_sided_p']:.8f} | "
            f"{values['bonferroni_p_five_comparisons']:.8f} |")
    if plan["profile"] != "smoke":
        supported = primary["mean_quantum_minus_classical"] < 0 and primary[
            "bonferroni_p_five_comparisons"] < 0.05
        rows += ["", ("新增24场在固定比较口径下支持量子特征的平均重建收益。" if supported else
                      "新增24场的平均差值与统计不确定性如上，不将其描述为已经证明的统计优势。")]
    rows += [
        "", "## 全部分组", "",
        "| 组 | C | Q | 降低% | 更低场数 | 逐组报告 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in descriptions:
        name = row["group"]
        rows.append(
            f"| {name} | {row['classical_mean']:.6f} | {row['quantum_mean']:.6f} | "
            f"{row['relative_reduction_of_means_percent']:.2f} | "
            f"{row['quantum_better_fields']}/{row['independent_fields']} | "
            f"[报告]({name}/analysis/REPORT.md) |")
    rows += [
        "", "## 口径与复核", "",
        f"- 全部{plan['expected_runs']}次训练完成并通过逐组哈希、选模、核矩阵和预测指标核验。",
        "- 每场先平均训练重复，再进行场级比较。变化条件复用第一组场种子，不按不同条件重复计数。",
        "- 四个复验组单列为描述性分解；五项预设比较同时报告原始p和Bonferroni调整p。",
        "- 区间是未校正的场级百分位bootstrap探索性区间，符号翻转依赖差值的对称/交换性假设。",
        "- 每组首场首次重复的初始与最佳权重重新自动微分核对，原始J、K、权重与日志完整保存。",
        "- 并行运行下训练耗时不作为独占资源成本基准；没有等参数、强周期基线或硬件量子加速结论。",
        "- 场内经典/量子比较共享数据；跨条件的误差变化不单独证明条件作用机制。",
        "- 这是逐场重新拟合的重建任务，不是零样本新场泛化。",
        "", "固定方案见[source/v3/ntk/SUPPLEMENT_PROTOCOL.md](source/v3/ntk/SUPPLEMENT_PROTOCOL.md)，"
        "完整配置见[plan.json](plan.json)，逐场结果见[fields.csv](fields.csv)。",
    ]
    (output / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def launch(output, workers, smoke):
    """先冻结方案再启动进程，全部组结束前不生成选择性的总体结论。"""
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "v3/ntk/runs"):
        raise ValueError("supplement output must be inside v3/ntk/runs")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": 1, "profile": "smoke" if smoke else "research",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "groups": groups(smoke), "epochs": 2 if smoke else 1000,
        "model_config": asdict(ModelConfig(features=2, hidden=6, feature_layers=1)
                               if smoke else ModelConfig()),
        "checkpoint_epochs": [0, 1, 2] if smoke else [0, 100, 300, 1000],
        "workers": workers, "expected_runs": 16 if smoke else 288,
        "primary": "all new replication fields; exclude historical and robustness fields",
        "multiple_comparisons": "Bonferroni, five planned comparisons",
        "source_sha256": source_snapshot(output),
    }
    write_json(output / "plan.json", plan)
    state = {"status": "running", "plan_sha256": digest(output / "plan.json"),
             "groups": {group["name"]: "pending" for group in plan["groups"]}}
    write_json(output / "status.json", state)
    # 每个子进程单线程；避免BLAS及Torch隐式线程竞争影响运行稳定性。
    env = {**os.environ, "PYTHONPATH": str(ROOT / "v3/src"), "PYTHONUNBUFFERED": "1"}
    env.update({name: "1" for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})

    def execute(spec):
        name = spec["name"]
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--output", str(output / name), "--worker-group", name]
        if smoke:
            command.append("--smoke")
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        return name

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, spec): spec["name"] for spec in plan["groups"]}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                state["groups"][name] = "completed"
                print(f"VERIFIED {name}", flush=True)
            except Exception as exc:
                state["groups"][name] = f"failed: {type(exc).__name__}: see {name}.log"
                print(f"FAILED {name}: see log", flush=True)
            write_json(output / "status.json", state)
    if any(value != "completed" for value in state["groups"].values()):
        state["status"] = "failed"
        write_json(output / "status.json", state)
        raise RuntimeError("some groups failed; preserve archives and inspect logs")
    try:
        aggregate(output, plan)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "status.json", state)
        raise
    state.update(status="completed", completed_runs=plan["expected_runs"],
                 completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "status.json", state)
    print(f"COMPLETED {plan['expected_runs']} fits; report: {output.relative_to(ROOT)}/REPORT.md",
          flush=True)


if __name__ == "__main__":
    # 内部worker参数只确定预设组，不提供按结果选种子或超参数的入口。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-group", choices=[group["name"] for group in groups()])
    args = parser.parse_args()
    if args.worker_group:
        worker(args.output, args.worker_group, args.smoke)
    else:
        launch(args.output, args.workers, args.smoke)
