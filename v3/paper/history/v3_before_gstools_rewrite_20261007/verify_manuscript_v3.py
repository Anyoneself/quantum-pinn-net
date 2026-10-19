"""Check manuscript links, references, table values and rendered figure geometry."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/manuscript_v3"
PAPER = ROOT / "paper/QUANTUM_DARCY_PINN_MANUSCRIPT_V3.md"
SUPPLEMENT = ROOT / "paper/materical/SUPPLEMENTARY_MATERIAL_V3.md"
SCRIPTS = Path(os.environ.get(
    "NATURE_FIGURE_SCRIPTS",
    str(Path.home() / ".codex/skills/nature-figure/scripts"),
))
LABELS = ["C", "C+RBA", "C+MC", "C+RBA+MC", "Q", "Q+RBA", "Q+MC", "Q+RBA+MC"]


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def run_audit(script, args, destination):
    completed = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                               capture_output=True, text=True, check=False)
    result = {"returncode": completed.returncode, "stderr": completed.stderr}
    try:
        result["output"] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result["output"] = completed.stdout
    write(destination, result)
    assert completed.returncode == 0, (script, args, completed.stdout, completed.stderr)
    return result


def main():
    text = PAPER.read_text()
    body, refs = text.split("## 参考文献\n", 1)
    entries = re.findall(r"^\[(\d+)\] (.+)$", refs, re.M)
    assert [int(i) for i, _ in entries] == list(range(1, 32))
    cited = set()
    prose = re.sub(r"\\\(.*?\\\)|\$\$.*?\$\$", "", body, flags=re.S)
    for cluster in re.findall(r"\[(\d+(?:[-,]\d+)*)\]", prose):
        for part in cluster.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cited.update(range(start, end+1))
            else:
                cited.add(int(part))
    assert cited == set(range(1, 32)), cited
    dois = re.findall(r"DOI:\s*(\S+)\.", refs)
    assert len(dois) == len(set(d.lower() for d in dois))
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    assert len([p for p in links if p.endswith(".png")]) == 5
    for link in links:
        assert (PAPER.parent / link).resolve().exists(), link

    summary = load(ROOT / "v3/experiments/quantum_pinn_v3_initial_20261007/summary.json")
    methods = list(summary["methods"])
    table = body.split("**表1 ", 1)[1].split("\n注：", 1)[0]
    checked_cells = 0
    for label, method in zip(LABELS, methods):
        row = next(line for line in table.splitlines() if line.startswith(f"| {label} |"))
        cells = [c.strip().replace("**", "") for c in row.strip("|").split("|")]
        for cell, metric, decimals in zip(cells[1:],
                                         ("k_relative_l2", "pressure_rmse", "pde_residual_rmse",
                                          "training_seconds"), (6, 6, 6, 3)):
            m = summary["methods"][method][metric]
            assert cell == f"{m['mean']:.{decimals}f}±{m['sd']:.{decimals}f}", (label, metric)
            checked_cells += 1
    evidence = load(OUT / "evidence.json")
    for comparison, token in (
        ("quantum_plain_vs_classical_plain", "3.09%"),
        ("quantum_mc_vs_classical_mc", "4.11%"),
        ("quantum_mc_vs_quantum_plain", "0.49%"),
    ):
        value = evidence["comparisons"][comparison]["relative_reduction_percent"]
        assert f"{value:.2f}%" == token and token in body
    assert "七次优于" in body and "一次改善、五次保持相同、三次小幅变差" in body
    assert "没有为本文追加训练" in body

    auxiliary = SUPPLEMENT.read_text().split("## S3 完整辅助指标", 1)[1].split("逐次结果", 1)[0]
    for label, method in zip(LABELS, methods):
        row = next(line for line in auxiliary.splitlines() if line.startswith(f"| {label} |"))
        for metric in ("log_k_rmse", "downstream_flux_relative_error"):
            m = summary["methods"][method][metric]
            assert f"{m['mean']:.6f}±{m['sd']:.6f}" in row
            checked_cells += 1
    # Architecture drawing delegates exports to the existing analysis module.
    # Audit the two sources together so callback exports remain visible.
    source_paths = [
        ROOT / "gen-figures/analyze_manuscript_v3.py",
        ROOT / "gen-figures/draw_v3_architecture.py",
    ]
    with tempfile.TemporaryDirectory(prefix="v3-figure-preflight-") as temporary:
        combined = Path(temporary) / "figure_sources.py"
        combined.write_text("\n\n".join(path.read_text() for path in source_paths))
        source = run_audit("validate_figure.py", [str(combined), "--json"],
                           OUT / "source-preflight.json")
    source["audited_sources"] = [str(path.relative_to(ROOT)) for path in source_paths]
    write(OUT / "source-preflight.json", source)
    figures = []
    for pdf in sorted(OUT.glob("fig[0-9][0-9]_*.pdf")):
        if ".collision" in pdf.name:
            continue
        name = pdf.stem
        run_audit("audit_pdf_text.py", [str(pdf), "--min-pt", "5", "--json"],
                  OUT / f"{name}.text-audit.json")
        collision = subprocess.run([
            sys.executable, str(SCRIPTS / "audit_figure_collisions.py"), str(pdf),
            "--json-out", str(OUT / f"{name}.collision-audit.json"),
        ], capture_output=True, text=True, check=False)
        assert collision.returncode == 0, collision.stdout + collision.stderr
        assert (OUT / f"{name}.alignment.json").exists()
        figures.append(name)
    report = {
        "status": "passed", "citations": len(cited), "embedded_figures": len(figures),
        "checked_table_cells": checked_cells,
        "body_chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", body)),
        "main_sha256": hashlib.sha256(PAPER.read_bytes()).hexdigest(),
        "supplement_sha256": hashlib.sha256(SUPPLEMENT.read_bytes()).hexdigest(),
        "figures": figures, "new_training": False,
        "scope": "links, reference numbering, DOI duplicates, selected table cells, key effects, figure audits",
        "limits": "Not peer review, literature comprehension validation, or independent training reproduction.",
        "source_preflight_returncode": source["returncode"],
    }
    write(OUT / "verification.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
