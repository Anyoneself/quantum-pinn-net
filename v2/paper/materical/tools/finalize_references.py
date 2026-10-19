"""Normalize citation order and render the manually curated reading ledger."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
MATERIAL = ROOT / "v2/paper/materical"
PAPER = ROOT / "v2/paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md"
CITE = re.compile(r"\[(\d+(?:[-,]\d+)*)\]")
PROTECTED = re.compile(r"(\$\$.*?\$\$|\\\(.*?\\\)|```.*?```)", re.DOTALL)


def expand(value):
    numbers = []
    for part in value.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            numbers.extend(range(lo, hi + 1))
        else:
            numbers.append(int(part))
    return numbers


def compact(numbers):
    numbers = sorted(set(numbers))
    parts = []
    start = last = numbers[0]
    for number in numbers[1:] + [None]:
        if number is not None and number == last + 1:
            last = number
            continue
        parts.append(str(start) if start == last else f"{start}-{last}")
        start = last = number
    return "[" + ",".join(parts) + "]"


def main():
    text = PAPER.read_text()
    body, bibliography = text.split("## 参考文献", 1)
    entries = {int(n): entry for n, entry in re.findall(r"(?m)^\[(\d+)\] (.+)$", bibliography)}
    assert len(entries) >= 30
    segments = PROTECTED.split(body)
    order = []
    for segment in segments[::2]:
        for match in CITE.finditer(segment):
            for number in expand(match[1]):
                assert number in entries, f"Unknown citation: {number}"
                if number not in order:
                    order.append(number)
    assert set(order) == set(entries), f"Uncited references: {set(entries) - set(order)}"
    remap = {old: new for new, old in enumerate(order, 1)}
    for i in range(0, len(segments), 2):
        segments[i] = CITE.sub(lambda m: compact([remap[n] for n in expand(m[1])]), segments[i])
    final_entries = {remap[old]: entries[old] for old in order}
    paper_text = "".join(segments) + "## 参考文献\n\n"
    paper_text += "\n\n".join(f"[{n}] {entry}" for n, entry in final_entries.items()) + "\n"

    notes = json.loads((MATERIAL / "literature-reading.json").read_text())
    assert len(notes) == len(entries) == len({n["key"] for n in notes})
    verified = []
    for note in notes:
        hits = [(n, entry) for n, entry in final_entries.items()
                if note["match"].casefold() in entry.casefold()]
        assert len(hits) == 1, (note["key"], hits)
        number, entry = hits[0]
        record_path = MATERIAL / "literature_sources" / f"{note['key']}.json"
        record = json.loads(record_path.read_text())
        source_path = ROOT / record["text_file"]
        assert source_path.is_file(), source_path
        assert "This is a preview of subscription content" not in source_path.read_text()
        metadata_sources = [os.path.relpath(record_path, MATERIAL)]
        if note["key"] in ("deepsdf", "causal", "wight"):
            metadata_sources.append(f"literature_sources/{note['key']}_candidates.json")
        for suffix in ("_proceedings.json", "_bibtex.txt", "_first_page.txt"):
            path = MATERIAL / "literature_sources" / f"{note['key']}{suffix}"
            if path.exists():
                # OpenReview's browser challenge is not a metadata verification.
                if "Verifying your browser" not in path.read_text():
                    metadata_sources.append(os.path.relpath(path, MATERIAL))
        verified.append({**note, "number": number, "bibliography": entry,
                         "reading_source": record["reading_source"],
                         "local_text": os.path.relpath(source_path, MATERIAL),
                         "accessed": record["accessed"], "metadata_sources": metadata_sources,
                         "status": "verified_and_relevant_sections_read"})
    verified.sort(key=lambda item: item["number"])

    lines = ["# 文献核验与成果阅读记录", "",
             "本记录对应[当前主稿](../PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md)。共32篇去重研究论文，均已核对题录、阅读与引用判断有关的原文内容，并在主稿中实际引用。公开作者稿与正式发表版本按同一研究计数，不重复累计。阅读范围是下列实际章节、结果和图表说明，不表示逐字阅读了每篇全部附录。", "",
             "核验来源包括Crossref、PMLR、NeurIPS、CVF、arXiv，以及已保存的两篇中文正式PDF。Crossref没有返回的中文题录，以原文首页核对。仅能访问摘要的旧候选文献不计入本次32篇；检索结果长度不作为全文判据。", "",
             "文献原文与提取文本只用于阅读核对；以下成果说明由本次阅读概括。网络获取没有新增成批PDF文件。来源版本差异和适用边界逐条说明。", ""]
    for item in verified:
        lines.extend([f"## [{item['number']}] {item['key']}", "", item["bibliography"], "",
                      f"- 阅读来源：{item['reading_source']}；[本地提取文本]({item['local_text']})。",
                      f"- 核验记录：" + "；".join(f"[来源{i + 1}]({p})" for i, p in enumerate(item["metadata_sources"])) + "。",
                      f"- 访问日期：{item['accessed'][:10]}。",
                      f"- 实际阅读范围：{item['sections']}",
                      f"- 研究成果：{item['finding']}",
                      f"- 本文引用用途：{item['use']}",
                      f"- 边界与版本说明：{item['boundary']}",
                      "- 状态：题录已核对；相关原文已读；已引用。", ""])
    lines.extend(["## 未计入的候选", "",
                  "- Dealing with spatial heterogeneity：本次出版商页面只有订阅预览。保留检索痕迹，不把摘要和参考文献列表视为全文，已从主稿参考文献移除。",
                  "- An adaptive inverse-distance weighting spatial interpolation technique：题录可核验，但本次没有得到可读正文，已从主稿参考文献移除。IDW作为本文明确给出的距离加权计算方法说明，不据此虚构原文阅读。",
                  "- 初始hard候选误将另一篇逆向设计论文的DOI与距离函数论文配对，已按实际PDF首页及Crossref改正为Sukumar和Srivastava的研究，错误配对不进入主稿。", ""])
    PAPER.write_text(paper_text)
    (MATERIAL / "literature-notes.md").write_text("\n".join(lines))
    report = {"listed": len(entries), "unique_works": len(verified),
              "relevant_full_text_read": len(verified), "actually_cited": len(order),
              "reading_scope": "Relevant original methods/results/discussion, including disclosed author-manuscript versions",
              "items": verified}
    (MATERIAL / "literature-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
