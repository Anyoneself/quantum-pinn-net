"""Run source and rendered-PDF checks for the five current V3 result figures."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "v3/paper/figures/manuscript_v3_gstools"
SCRIPTS = Path(os.environ.get(
    "NATURE_FIGURE_SCRIPTS",
    str(Path.home() / ".codex/skills/nature-figure/scripts"),
))


def save(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def execute(name, arguments):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, arguments)],
                          text=True, capture_output=True, check=False)


def main():
    preflight = execute("validate_figure.py", [
        ROOT / "v3/src/scripts/analyze_v3_gstools_results.py", "--json",
    ])
    source_report = json.loads(preflight.stdout)
    save(OUT / "results_source_preflight.json", source_report)
    failed = preflight.returncode != 0
    rows = []
    manifest = json.loads((OUT / "result_figures.json").read_text())
    for figure in manifest["figures"]:
        name = figure["name"]
        pdf = OUT / f"{name}.pdf"
        text = execute("audit_pdf_text.py", [pdf, "--min-pt", "5", "--json"])
        save(OUT / f"{name}.text-audit.json", json.loads(text.stdout))
        collision = execute("audit_figure_collisions.py", [
            pdf, "--json-out", OUT / f"{name}.collision-audit.json",
        ])
        failed |= text.returncode != 0 or collision.returncode != 0
        rows.append({
            "name": name, "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "text_exit_code": text.returncode,
            "collision_exit_code": collision.returncode,
            "collision_output": collision.stdout.strip(),
            "stderr": text.stderr + collision.stderr,
        })
    report = {"status": "needs_repair" if failed else "automated_checks_passed",
              "source_preflight": source_report["summary"], "figures": rows,
              "limits": "Automated checks do not replace visual/semantic review."}
    save(OUT / "results_render_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
