#!/usr/bin/env python3
"""Run submission-facing static checks on the phase-periodic manuscript."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EXPECTED_CHAPTERS = [
    "## 1 引言",
    "## 2 理论基础与方法设计",
    "## 3 实验与结果分析",
    "## 4 总结与展望",
]

EXPECTED_SECTIONS = [
    "### 1.1 研究背景与意义",
    "### 1.2 国内外研究现状",
    "### 1.3 研究内容与论文结构",
    "### 2.1 相关理论基础",
    "### 2.2 本文方法或模型设计",
    "### 2.3 本章小结",
    "### 3.1 实验设置",
    "### 3.2 实验结果与分析",
    "### 3.3 本章小结",
    "### 4.1 研究工作总结",
    "### 4.2 局限性与未来展望",
]

FORBIDDEN_PATTERNS = {
    "按轮次叙述": r"第[一二三四五六七八九十百0-9]+轮",
    "筛选种子": r"筛选种子",
    "比例式流水账": r"70\s*%.*10\s*%.*20\s*%",
    "源码对照措辞": r"与源码|源码中|源码实现",
    "项目实现措辞": r"项目实现",
    "内部审计措辞": r"保存的预测数组|审计日志",
    "旧任务称谓": r"已登记场",
}

FORBIDDEN_MODEL_ALIASES = [
    "Baseline PINN",
    "Quantum Feature PINN",
    "RBA-QPINN",
    "量子特征 PINN",
]

PLACEHOLDER_PATTERNS = {
    "作者或单位信息未填写": r"【待补】",
    "公开链接未填写": r"【公开链接待补】",
    "作者确认尚未完成": r"【需全体作者确认】",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate structure, numbering, citations, wording, and figure paths."
    )
    parser.add_argument(
        "manuscript",
        nargs="?",
        type=Path,
        default=Path("v2/paper/PHASE_PERIODIC_ENGINEERING_MANUSCRIPT_V2.md"),
    )
    return parser.parse_args()


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def ordered_unique(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def expand_citation_group(group: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"[,，、]\s*", group):
        bounds = [int(value) for value in re.findall(r"\d+", part)]
        if len(bounds) == 2 and re.search(r"[–—-]", part):
            numbers.extend(range(bounds[0], bounds[1] + 1))
        elif len(bounds) == 1:
            numbers.append(bounds[0])
    return numbers


def check_contiguous(
    label: str,
    values: list[int],
    errors: list[str],
) -> None:
    unique = ordered_unique(values)
    expected = list(range(1, max(unique) + 1)) if unique else []
    if unique != expected or len(values) != len(unique):
        errors.append(f"{label}编号异常：读取到 {values}")


def main() -> int:
    args = parse_args()
    manuscript = args.manuscript.resolve()
    if not manuscript.is_file():
        print(f"[FAIL] 稿件不存在：{manuscript}")
        return 1

    text = manuscript.read_text(encoding="utf-8")
    body, separator, references_text = text.partition("## 参考文献")
    errors: list[str] = []
    warnings: list[str] = []

    chapter_headings = re.findall(r"^##\s+[1-4]\s+.+$", body, flags=re.MULTILINE)
    if chapter_headings != EXPECTED_CHAPTERS:
        errors.append(f"四章结构不匹配：{chapter_headings}")

    section_headings = re.findall(r"^###\s+[1-4]\.\d+\s+.+$", body, flags=re.MULTILINE)
    if section_headings != EXPECTED_SECTIONS:
        errors.append(f"一级小节结构不匹配：{section_headings}")

    formulas = [int(value) for value in re.findall(r"\\tag\{(\d+)\}", body)]
    tables = [int(value) for value in re.findall(r"\*\*表\s*(\d+)", body)]
    figures = [int(value) for value in re.findall(r"\*\*图\s*(\d+)", body)]
    check_contiguous("公式", formulas, errors)
    check_contiguous("表", tables, errors)
    check_contiguous("图", figures, errors)

    if not separator:
        errors.append("缺少参考文献章节")
        reference_numbers: list[int] = []
    else:
        reference_numbers = [
            int(value)
            for value in re.findall(r"^\[(\d+)\]", references_text, flags=re.MULTILINE)
        ]
        check_contiguous("参考文献", reference_numbers, errors)

    citation_pattern = re.compile(
        r"\[([1-9]\d*(?:\s*[–—-]\s*[1-9]\d*)?"
        r"(?:\s*[,，、]\s*[1-9]\d*(?:\s*[–—-]\s*[1-9]\d*)?)*)\]"
    )
    citations: list[int] = []
    for match in citation_pattern.finditer(body):
        citations.extend(expand_citation_group(match.group(1)))
    first_seen = ordered_unique(citations)
    if first_seen != reference_numbers:
        errors.append(
            "引文首次出现顺序与参考文献不一致："
            f"首次出现 {first_seen}；文后表 {reference_numbers}"
        )

    for label, pattern in FORBIDDEN_PATTERNS.items():
        for match in re.finditer(pattern, body, flags=re.IGNORECASE | re.DOTALL):
            errors.append(
                f"{label}（第 {line_number(body, match.start())} 行）：{match.group(0)}"
            )

    for alias in FORBIDDEN_MODEL_ALIASES:
        for match in re.finditer(re.escape(alias), body, flags=re.IGNORECASE):
            errors.append(
                f"正文仍含旧模型名（第 {line_number(body, match.start())} 行）：{alias}"
            )

    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", body):
        target = match.group(1).strip()
        if re.match(r"(?:https?://|data:)", target):
            continue
        figure_path = (manuscript.parent / target).resolve()
        if not figure_path.is_file():
            errors.append(
                f"图文件不存在（第 {line_number(body, match.start())} 行）：{figure_path}"
            )

    for label, pattern in PLACEHOLDER_PATTERNS.items():
        matches = list(re.finditer(pattern, text))
        if matches:
            lines = ", ".join(str(line_number(text, match.start())) for match in matches)
            warnings.append(f"{label}：第 {lines} 行")

    if formulas:
        print(f"[PASS] 公式编号连续：1–{formulas[-1]}")
    if tables:
        print(f"[PASS] 表编号连续：1–{tables[-1]}")
    if figures:
        print(f"[PASS] 图编号连续：1–{figures[-1]}")
    if reference_numbers and first_seen == reference_numbers:
        print(f"[PASS] 引用及参考文献顺序一致：1–{reference_numbers[-1]}")
    if chapter_headings == EXPECTED_CHAPTERS and section_headings == EXPECTED_SECTIONS:
        print("[PASS] 四章结构及一级小节符合既定架构")

    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[FAIL] {error}")

    if errors:
        print(f"[FAIL] 静态门禁未通过：{len(errors)} 项错误，{len(warnings)} 项待补信息")
        return 1
    print(f"[PASS] 静态门禁通过；仍有 {len(warnings)} 类投稿信息待补")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
