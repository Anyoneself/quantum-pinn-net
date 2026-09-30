from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/manuscript_v2"
TOOLS = Path.home() / ".codex/skills/nature-figure/scripts"


def main():
    source = subprocess.run(
        [sys.executable, str(TOOLS / "validate_figure.py"),
         str(ROOT / "gen-figures/analyze_supplement_v2.py"), "--json"],
        capture_output=True, text=True, check=True,
    )
    (OUT / "source-audit.json").write_text(source.stdout)
    reports = []
    for pdf in sorted(OUT.glob("fig*.pdf")):
        collision_path = pdf.with_suffix(".collision-audit.json")
        proc = subprocess.run(
            [sys.executable, str(TOOLS / "audit_figure_collisions.py"), str(pdf),
             "--json-out", str(collision_path)], capture_output=True, text=True,
        )
        glyphs = subprocess.run(
            [sys.executable, str(TOOLS / "audit_pdf_text.py"), str(pdf), "--min-pt", "5", "--json"],
            capture_output=True, text=True,
        )
        pdf.with_suffix(".glyph-audit.json").write_text(glyphs.stdout)
        item = {"figure": pdf.name, "collision_exit": proc.returncode, "glyph_exit": glyphs.returncode}
        if proc.returncode or glyphs.returncode:
            print(proc.stdout, glyphs.stdout, proc.stderr, glyphs.stderr)
        reports.append(item)
    (OUT / "figure-audit-summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    if len(reports) != 8 or any(x["collision_exit"] or x["glyph_exit"] for x in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
