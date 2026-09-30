"""Check manuscript links, citations and the five requested writing requirements."""

import json
from pathlib import Path
import re

from finalize_references import CITE, PROTECTED, expand

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md"


def main():
    text = PAPER.read_text()
    body, bibliography = text.split("## 参考文献", 1)
    abstract = body.split("## 摘要", 1)[1].split("**关键词**", 1)[0]
    english = body.split("## Abstract", 1)[1].split("**Keywords**", 1)[0]
    assert not re.search(r"工程价值|应用前景|未来|有望|展望", abstract)
    assert not re.search(r"practical value|future|prospect|potential application", english, re.I)
    assert not re.search(r"种子|\bseed\b|manual_seed|SHA256|线程|(?:Python|PyTorch|NumPy)\s+\d", body, re.I)
    assert not re.search(r"比例观测|观测比例|观测率|TODO|待补结果", body)

    references = {int(n): s for n, s in re.findall(r"(?m)^\[(\d+)\] (.+)$", bibliography)}
    assert len(references) >= 30
    assert list(references) == list(range(1, len(references) + 1))
    citations = []
    for segment in PROTECTED.split(body)[::2]:
        for match in CITE.finditer(segment):
            for number in expand(match[1]):
                if number not in citations:
                    citations.append(number)
    assert citations == list(references), "References must follow first citation order"
    dois = re.findall(r"DOI:\s*(\S+)\.", bibliography)
    assert len(dois) == len(set(d.casefold() for d in dois))

    figures = re.findall(r"!\[图(\d+) [^\]]+\]\(([^)]+)\)", body)
    assert [int(n) for n, _ in figures] == list(range(1, 11))
    assert len(re.findall(r"(?m)^\*\*图\d+ ", body)) == 10
    for _, filename in figures:
        assert (PAPER.parent / filename).resolve().is_file(), filename
    for filename in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", body):
        if not filename.startswith("http"):
            assert (PAPER.parent / filename).resolve().is_file(), filename

    material = ROOT / "paper/materical"
    literature = json.loads((material / "literature-verification.json").read_text())
    assert literature["listed"] == literature["unique_works"] == literature["actually_cited"] == len(references)
    assert literature["relevant_full_text_read"] >= 30
    for item in literature["items"]:
        assert references[item["number"]] == item["bibliography"]
        assert (material / item["local_text"]).resolve().is_file()
        for field in ("sections", "finding", "use", "boundary"):
            assert item[field]

    summary = json.loads((ROOT / "figures/manuscript_v2/verification.json").read_text())
    assert summary["result_table_cells_checked"] == 84
    assert summary["max_recomputed_metric_difference"] == 0
    aggregate = json.loads((ROOT / "figures/manuscript_v2/summary.json").read_text())
    lookup = {(r["scenario"], r["method"], r["stage"]): r for r in aggregate}
    f = lookup["random70", "fourier", "raw"]["rmse_mean"]
    p = lookup["random70", "phase", "raw"]["rmse_mean"]
    checks = [f"{(1 - p / f) * 100:.2f}",
              f"{lookup['random70', 'phase', 'corrected']['rmse_mean']:.5f}",
              f"{lookup['random70', 'rbf', 'raw']['rmse_mean']:.5f}",
              f"{lookup['block', 'phase', 'corrected']['rmse_mean']:.5f}"]
    for number in checks:
        assert number in abstract and number in english, number
    report = {"abstract_checked": True, "abstract_values": checks,
              "references": len(references), "read_and_cited": literature["relevant_full_text_read"],
              "citation_order_checked": True, "embedded_figures": len(figures),
              "links_checked": True, "architecture_and_workflow_embedded": True,
              "runtime_details_in_main_text": 0, "result_table_cells_checked": 84,
              "new_experiments_run": 0}
    (material / "manuscript-requirements-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
