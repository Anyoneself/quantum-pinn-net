"""Check the current illustrated V3 manuscript against its frozen GSTools archive."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / "v3/paper/QUANTUM_DARCY_PINN_MANUSCRIPT_V3.md"
SUPPLEMENT = ROOT / "v3/paper/materical/SUPPLEMENTARY_MATERIAL_V3.md"
OUTLINE = ROOT / "v3/paper/V3_MANUSCRIPT_OUTLINE.md"
ARCHIVE = ROOT / "v3/experiments/quantum_pinn_v3_gstools_konly_20261007"
OUTPUT = ROOT / "v3/paper/materical/MANUSCRIPT_V3_VERIFICATION.json"
FIGURES = ROOT / "v3/paper/figures/manuscript_v3_gstools"
LABELS = ["C", "C+RBA", "C+MC", "C+RBA+MC",
          "Q", "Q+RBA", "Q+MC", "Q+RBA+MC"]
METHODS = [f"{kind}_{strategy}" for kind in ("classical", "quantum")
           for strategy in ("plain", "rba", "mc", "rba_mc")]


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def table(text, title):
    section = text.split(f"**{title} ", 1)[1]
    rows = {}
    started = False
    for line in section.splitlines():
        if line.startswith("|"):
            started = True
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            rows[cells[0]] = cells[1:]
        elif started:
            break
    require(bool(rows), f"empty table: {title}")
    return rows


def verify_result_figures(manifest):
    import numpy as np

    metadata = load(FIGURES / "result_figures.json")
    require(metadata["manifest_sha256"] == digest(ARCHIVE / "manifest.json"),
            "result figures use a different archive")
    # 图件保留原脚本哈希；迁移后的源码字节不同，不能用新路径冒充原绘制版本。
    require(re.fullmatch(r"[0-9a-f]{64}", metadata["script_sha256"]) is not None,
            "invalid historical plotting source hash")
    require(len(metadata["figures"]) == 5, "result figure count")
    for relative, expected in metadata["artifact_sha256"].items():
        path = (FIGURES / relative).resolve()
        require(path.is_relative_to(FIGURES), "unsafe figure path")
        require(digest(path) == expected, f"figure artifact changed: {relative}")
    audit = load(FIGURES / "results_render_audit.json")
    require(audit["status"] == "automated_checks_passed", "render audits failed")
    for item in audit["figures"]:
        require(item["pdf_sha256"] == digest(FIGURES / f"{item['name']}.pdf"),
                "render audit predates PDF")

    def rows(name):
        with (FIGURES / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def equal(actual, expected, label):
        require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-14),
                f"figure source differs: {label}")

    metrics = rows("result_run_metrics.csv")
    require(len(metrics) == 72, "figure run source count")
    results, histories = {}, {}
    for field in manifest["field_seeds"]:
        for seed in manifest["seeds"]:
            for method in METHODS:
                folder = ARCHIVE / f"field_{field}/seed_{seed}/{method}"
                key = (field, seed, method)
                results[key] = load(folder / "result.json")
                histories[key] = {r["epoch"]: r for r in load(folder / "history.json")}
    seen = set()
    for row in metrics:
        key = (int(row["field"]), int(row["seed"]), row["method"])
        require(key not in seen, "duplicate figure run")
        seen.add(key)
        for metric, value in row.items():
            if metric not in ("field", "seed", "method"):
                equal(value, results[key][metric], metric)
    paired = rows("paired_effects_source.csv")
    require(len(paired) == 36, "paired source count")
    for row in paired:
        key = (int(row["field"]), int(row["seed"]))
        delta = (results[*key, row["candidate"]]["k_relative_l2"]
                 - results[*key, row["baseline"]]["k_relative_l2"])
        equal(row["delta_k_relative_l2"], delta, "paired difference")

    spatial = rows("spatial_results_source.csv")
    n = manifest["data_config"]["grid"]
    seed = manifest["seeds"][0]
    require(len(spatial) == n * n * len(manifest["field_seeds"]), "spatial row count")
    spatial_lookup = {}
    for field in manifest["field_seeds"]:
        with np.load(ARCHIVE / f"field_{field}/reference.npz") as ref:
            xy, truth = ref["xy"], np.exp(ref["log_k"])
        with np.load(ARCHIVE / f"field_{field}/seed_{seed}/observations.npz") as obs:
            splits = {int(i): label for label in ("train", "validation", "test")
                      for i in obs[f"k_{label}_indices"]}
        prediction = {}
        for method in ("classical_plain", "quantum_plain", "quantum_rba_mc"):
            with np.load(ARCHIVE / f"field_{field}/seed_{seed}/{method}/predictions.npz") as arr:
                prediction[method] = np.exp(arr["log_k"])
        for row in [r for r in spatial if int(r["field"]) == field]:
            index = int(row["index"])
            require((field, index) not in spatial_lookup, "duplicate spatial position")
            spatial_lookup[field, index] = row
            require(int(row["seed"]) == seed and row["split"] == splits[index],
                    "spatial selection or test mask differs")
            for key, expected in (("x", xy[index, 0]), ("y", xy[index, 1]),
                                  ("reference_k", truth[index])):
                equal(row[key], expected, key)
            for method in prediction:
                equal(row[method], prediction[method][index], method)
    profiles = rows("profile_results_source.csv")
    expected_profiles = [row for row in spatial if n * (n // 2)
                         <= int(row["index"]) < n * (n // 2 + 1)]
    require(profiles == expected_profiles, "profile selection differs")

    training = rows("training_results_source.csv")
    epochs = manifest["train_config"]["epochs"]
    require(len(training) == 72 * (2 * epochs + 1), "training source count")
    grouped = {}
    seen = set()
    for row in training:
        field, seed, epoch = int(row["field"]), int(row["seed"]), int(row["epoch"])
        metric = ("validation" if row["metric"] == "validation_loss"
                  else "physics_unweighted_mse_pre_step")
        key = (field, seed, row["method"])
        full_key = (*key, metric, epoch)
        require(full_key not in seen, "duplicate training observation")
        seen.add(full_key)
        expected = histories[key][epoch][metric]
        if metric != "validation":
            expected = math.sqrt(expected)
        equal(row["value"], expected, "training value")
        grouped.setdefault((row["method"], metric, epoch), []).append(expected)
    trajectory_summary = rows("training_results_summary.csv")
    require(len(trajectory_summary) == 8 * (2 * epochs + 1), "trajectory summary count")
    for row in trajectory_summary:
        values = grouped[row["method"], row["metric"], int(row["epoch"])]
        require(len(values) == 9, "incomplete training aggregate")
        for name, value in zip(("q25", "median", "q75"), np.quantile(values, [0.25, 0.5, 0.75])):
            equal(row[name], value, "trajectory quantile")
    physical = rows("physics_cost_summary.csv")
    require(len(physical) == 16, "physical/cost summary count")
    for row in physical:
        values = [r[row["metric"]] for key, r in results.items() if key[2] == row["method"]]
        equal(row["mean"], statistics.mean(values), "physical/cost mean")
        equal(row["sample_sd"], statistics.stdev(values), "physical/cost SD")
    return {
        "figures": len(metadata["figures"]),
        "artifacts": len(metadata["artifact_sha256"]),
        "plot_source_check": "historical hash retained; relocated source not byte-identical",
        "csv_rows_checked": sum(map(len, (metrics, paired, spatial, profiles,
                                         training, trajectory_summary, physical))),
        "render_audits": audit["status"],
    }


def verify_budget_extension(body):
    folder = ROOT / "v3/paper/figures/manuscript_v3_budget5000"
    # 按稿件目录解析图像引用，避免相对路径导致预算核验被跳过。
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)
    if folder / "fig04_spatial_maps.png" not in {
        (PAPER.parent / link).resolve() for link in images
    }:
        return None
    checked = subprocess.run(
        [sys.executable, "-B", str(ROOT / "v3/src/scripts/analyze_v3_reconstruction_budget.py"),
         "--verify-only"],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(checked.stdout)
    require(report["status"] == "passed" and report["runs"] == 6, "budget verification failed")
    with (folder / "budget_comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for method in ("classical_plain", "quantum_plain"):
        selected = [row for row in rows if row["method"] == method]
        before = statistics.mean(float(row["old_k_relative_l2"]) for row in selected)
        after = statistics.mean(float(row["new_k_relative_l2"]) for row in selected)
        for number in (f"{before:.6f}", f"{after:.6f}", f"{100*(1-after/before):.2f}%"):
            require(number in body, f"budget summary missing or inconsistent: {number}")
    field2 = next(row for row in rows if row["method"] == "quantum_plain" and row["field"] == "22")
    for key in ("old_k_relative_l2", "new_k_relative_l2"):
        require(f"{float(field2[key]):.6f}" in body, "budget field-2 qualification missing")
    return report


def main():
    text, supplement = PAPER.read_text(), SUPPLEMENT.read_text()
    body, bibliography = text.split("## 参考文献\n", 1)
    manifest, summary = load(ARCHIVE / "manifest.json"), load(ARCHIVE / "summary.json")
    require(manifest["status"] == "completed", "archive incomplete")
    require(manifest["data_config"]["field_generator"] == "gstools", "wrong generator")
    require(manifest["data_config"]["p_train"] == 0, "pressure labels enabled")
    require(manifest["loss_weights"]["pressure"] == 0, "pressure loss enabled")
    for prefix, mapping in (("source", manifest["source_sha256"]),
                            ("", manifest["artifact_sha256"])):
        for relative, expected in mapping.items():
            path = (ARCHIVE / prefix / relative).resolve()
            require(path.is_relative_to(ARCHIVE.resolve()), f"unsafe path: {relative}")
            require(digest(path) == expected, f"archive hash mismatch: {relative}")

    headings = re.findall(r"^#{2,3} ([0-4](?:\.\d+)? .+)$", body, re.M)
    expected = re.findall(r"^#{3,4} ([0-4](?:\.\d+)? .+)$", OUTLINE.read_text(), re.M)
    require(headings == expected, "manuscript headings differ from the approved outline")
    require([int(n) for n in re.findall(r"\\tag\{(\d+)\}", body)]
            == list(range(1, 20)), "equation numbering")
    require(re.findall(r"^\*\*表(\d+) ", body, re.M)
            == [str(n) for n in range(1, 7)], "table numbering")
    entries = re.findall(r"^\[(\d+)\] (.+)$", bibliography, re.M)
    require([int(n) for n, _ in entries] == list(range(1, 34)), "reference numbering")
    cited = set()
    prose = re.sub(r"\$\$.*?\$\$|\\\(.*?\\\)|```.*?```", "", body, flags=re.S)
    for cluster in re.findall(r"\[(\d+(?:[-,]\d+)*)\]", prose):
        for part in cluster.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cited.update(range(start, end + 1))
            else:
                cited.add(int(part))
    require(cited == set(range(1, 34)), f"uncited or unknown references: {cited}")
    dois = re.findall(r"DOI:\s*(\S+)\.", bibliography)
    require(len(dois) == len(set(d.lower() for d in dois)), "duplicate DOI")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body)
    require(len(images) == 7 and "```text" not in body, "expected seven embedded figures")
    require(re.findall(r"^\*\*图(\d+) ", body, re.M) == [str(i) for i in range(1, 8)],
            "figure numbering")
    image_records = []
    for link in images:
        path = (PAPER.parent / link).resolve()
        header = path.read_bytes()[:24]
        require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR",
                f"invalid PNG header: {path}")
        width, height = struct.unpack(">II", header[16:24])
        require(min(width, height) >= 1000, f"undersized diagram: {path}")
        image_records.append({"path": str(path.relative_to(ROOT)), "width": width,
                              "height": height, "sha256": digest(path)})
    link_count = 0
    for path, content in ((PAPER, text), (SUPPLEMENT, supplement)):
        require(content.count("```") % 2 == 0, f"unbalanced code fence: {path.name}")
        require(all(line == line.rstrip() for line in content.splitlines()),
                f"trailing whitespace: {path.name}")
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            require(not link.startswith(("https://", "http://")), "unexpected remote link")
            # 文档本地引用须可随仓库迁移，禁止绝对路径及文件协议。
            require(not Path(link).is_absolute() and not link.startswith("file:"),
                    f"use document-relative local links: {link}")
            require((path.parent / link).resolve().exists(), f"missing link: {link}")
            link_count += 1
    for stale in ("0.102443", "0.105710", "3.09%", "4.11%",
                  "figures/manuscript_v3/", r"\mathcal L_p", r"\lambda_p"):
        require(stale not in body, f"historical content: {stale}")

    records = {}
    for method in METHODS:
        records[method] = [
            load(ARCHIVE / f"field_{field}" / f"seed_{seed}" / method / "result.json")
            for field in manifest["field_seeds"] for seed in manifest["seeds"]
        ]
        for metric in ("k_relative_l2", "log_k_rmse", "pressure_rmse",
                       "pde_residual_rmse", "downstream_flux_relative_error",
                       "training_seconds"):
            values = [row[metric] for row in records[method]]
            for key, computed in (("mean", statistics.mean(values)),
                                  ("sd", statistics.stdev(values))):
                require(math.isclose(computed, summary["methods"][method][metric][key],
                                     rel_tol=1e-12, abs_tol=1e-14),
                        f"summary differs from run records: {method}, {metric}, {key}")

    checked_cells = 0
    for title, metrics in (("表3", ("k_relative_l2", "log_k_rmse")),
                           ("表6", ("pressure_rmse", "pde_residual_rmse",
                                    "downstream_flux_relative_error", "training_seconds"))):
        rows = table(body, title)
        for label, method in zip(LABELS, METHODS):
            expected_cells = []
            for metric in metrics:
                item = summary["methods"][method][metric]
                expected_cells.append(f"{item['mean']:.6f}±{item['sd']:.6f}")
            require(rows[label] == expected_cells, f"{title}: {label}")
            checked_cells += len(metrics)
    rows = table(body, "表4")
    for index, field in enumerate(manifest["field_seeds"], 1):
        expected_cells = [
            f"{summary['methods'][method]['field_mean_k_relative_l2'][str(field)]:.6f}"
            for method in ("classical_plain", *METHODS[4:])
        ]
        require(rows[f"场{index}"] == expected_cells, f"field table: {field}")
        checked_cells += len(expected_cells)

    screening_rows, checkpoint_rows = table(body, "表5"), table(supplement, "表S4")
    for label, method in zip(LABELS, METHODS):
        runs = records[method]
        reduced = sum(row["active_features_at_best"] < 4 for row in runs)
        final = sum(row["best_epoch"] == row["epochs"] for row in runs)
        require(checkpoint_rows[label] == [str(final), str(reduced)],
                f"checkpoint statistics: {method}")
        checked_cells += 2
        if "mc" in method:
            logs = [
                load(ARCHIVE / f"field_{row['field_seed']}" / f"seed_{row['seed']}"
                     / method / "screening.json") for row in runs
            ]
            accepted = sum(event["accepted"] for events in logs for event in events)
            elapsed = statistics.mean(row["screening_seconds"] for row in runs)
            require(screening_rows[label] == [str(accepted), str(reduced), f"{elapsed:.6f}"],
                    f"screening statistics: {method}")
            checked_cells += 3

    rows = table(supplement, "表S1")
    for field in manifest["field_seeds"]:
        item = load(ARCHIVE / f"field_{field}" / "reference.json")
        values = [
            f"{item['permeability_contrast']:.6f}",
            f"{item['coarse_fine_pressure_rmse']:.8f}",
            f"{item['coarse_fine_flux_relative_difference']:.8f}",
            f"{item['fine_solver_residual']:.6e}",
            f"{item['fine_flux_balance']:.6e}",
        ]
        require(rows[str(field)] == values, f"reference quality: {field}")
        checked_cells += len(values)

    def mean(method, metric="k_relative_l2"):
        return summary["methods"][method][metric]["mean"]

    comparisons = (
        ("Q 对 C", "quantum_plain", "classical_plain", "k_relative_l2", "9.00%"),
        ("Q+RBA 对 Q", "quantum_rba", "quantum_plain", "k_relative_l2", "0.27%"),
        ("Q+MC 对 Q", "quantum_mc", "quantum_plain", "k_relative_l2", None),
        ("Q+RBA+MC 对 Q", "quantum_rba_mc", "quantum_plain", "k_relative_l2", "0.29%"),
        ("Q+RBA+MC 对 Q+RBA", "quantum_rba_mc", "quantum_rba", "k_relative_l2", "0.01747%"),
        ("Q+RBA 对 Q", "quantum_rba", "quantum_plain", "pde_residual_rmse", "24.07%"),
    )
    # The same label appears for two metrics, so inspect raw rows instead of a label map.
    for label, candidate, baseline, metric, token in comparisons:
        reduction = 100 * (1 - mean(candidate, metric) / mean(baseline, metric))
        symbol = r"\(E_r\)" if metric == "pde_residual_rmse" else r"\(E_k\)"
        require(f"| {label} | {symbol} | {reduction:.5f}% |" in supplement,
                f"relative reduction: {label}, {metric}")
        if token:
            decimals = len(token.split(".")[1].rstrip("%"))
            require(f"{reduction:.{decimals}f}%" == token and token in body,
                    f"body effect: {label}")
        checked_cells += 1
    require(f"{mean('quantum_plain', 'training_seconds') / mean('classical_plain', 'training_seconds'):.2f}倍"
            in body, "time ratio")
    increase = 100 * (mean("quantum_mc", "training_seconds") / mean("quantum_plain", "training_seconds") - 1)
    require(f"{increase:.2f}%" in body, "MC time increase")
    for key, counts in (
        ("quantum_plain_vs_classical_plain", (7, 0, 2)),
        ("quantum_rba_vs_quantum_plain", (3, 0, 6)),
        ("quantum_mc_vs_quantum_plain", (0, 8, 1)),
        ("quantum_rba_mc_vs_quantum_plain", (3, 0, 6)),
    ):
        values = [r["candidate_minus_baseline"] for r in summary["paired_comparisons"][key]["runs"]]
        require(tuple(sum(predicate(x) for x in values) for predicate in
                      (lambda x: x < 0, lambda x: x == 0, lambda x: x > 0)) == counts,
                f"paired counts: {key}")

    figure_checks = verify_result_figures(manifest)
    budget_checks = verify_budget_extension(body)
    report = {
        "status": "passed",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_artifacts": len(manifest["artifact_sha256"]),
        "archive_source_files": len(manifest["source_sha256"]),
        "run_records": sum(len(rows) for rows in records.values()),
        "outline_headings_matched": len(headings),
        "references": len(entries), "equations": 19,
        "main_tables": 6, "checked_numeric_cells": checked_cells,
        "text_diagrams": 0, "embedded_images": len(images), "local_links": link_count,
        "images": image_records,
        "result_figure_checks": figure_checks,
        "budget_extension_checks": budget_checks,
        "body_chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", body)),
        "main_sha256": digest(PAPER), "supplement_sha256": digest(SUPPLEMENT),
        "outline_sha256": digest(OUTLINE),
        "new_training": budget_checks is not None,
        "scope": "archive hashes, run aggregation, outline, references, links, equations, specified numeric tables and effects, PNG headers/dimensions/hashes, result-figure CSV values and render-audit freshness",
        "limits": "Checker does not train models or recompute checkpoint derivatives; no full external bibliography re-audit, automated semantic audit, schematic collision audit or peer review.",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
