from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the manuscript's leakage-aware raw spatial comparison figure."
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=ROOT / "experiments" / "rba_qpinn_submission_v1" / "seed_42",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "v2/paper/figures/manuscript/raw_spatial_comparison.png",
    )
    return parser.parse_args()


def load_arrays(seed_dir: Path) -> dict[str, np.ndarray]:
    processed = seed_dir / "processed"
    arrays = {
        "target": np.load(processed / "fields.npy", allow_pickle=False).astype(np.float64),
        "train_mask_raw": np.load(
            processed / "train_point_mask.npy", allow_pickle=False
        ),
        "validation_mask_raw": np.load(
            processed / "validation_point_mask.npy", allow_pickle=False
        ),
        "test_mask_raw": np.load(
            processed / "test_point_mask.npy", allow_pickle=False
        ),
        "fourier": np.load(
            seed_dir / "runs" / "standard-pinn" / "raw_predicted_log_perm.npy",
            allow_pickle=False,
        ).astype(np.float64),
        "phase": np.load(
            seed_dir / "runs" / "quantum-feature" / "raw_predicted_log_perm.npy",
            allow_pickle=False,
        ).astype(np.float64),
    }
    target_shape = arrays["target"].shape
    if target_shape != (128, 32, 32):
        raise ValueError(f"Unexpected field shape in {seed_dir}: {target_shape}")
    for name in ["target", "fourier", "phase"]:
        if arrays[name].shape != target_shape:
            raise ValueError(f"Unexpected shape for {name}: {arrays[name].shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Non-finite values in {name}")
    mask_names = ["train_mask_raw", "validation_mask_raw", "test_mask_raw"]
    for name in mask_names:
        if arrays[name].shape != target_shape:
            raise ValueError(f"Unexpected shape for {name}: {arrays[name].shape}")
        if not np.isin(arrays[name], [0, 1]).all():
            raise ValueError(f"Non-binary values in {name}")
    coverage = sum(arrays[name].astype(np.uint8) for name in mask_names)
    if not np.all(coverage == 1):
        raise ValueError("Training, validation, and held-out masks are not disjoint and complete")
    arrays["train_mask"] = arrays.pop("train_mask_raw").astype(bool)
    arrays.pop("validation_mask_raw")
    arrays["test_mask"] = arrays.pop("test_mask_raw").astype(bool)
    return arrays


def rmse_by_field(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.empty(target.shape[0], dtype=np.float64)
    for index in range(target.shape[0]):
        error = prediction[index][mask[index]] - target[index][mask[index]]
        result[index] = np.sqrt(np.mean(error**2))
    return result


def select_quantile_fields(improvement: np.ndarray) -> list[tuple[float, int]]:
    selected: list[tuple[float, int]] = []
    used: set[int] = set()
    for quantile in (0.25, 0.50, 0.75):
        target = float(np.quantile(improvement, quantile))
        for index in np.argsort(np.abs(improvement - target)):
            candidate = int(index)
            if candidate not in used:
                used.add(candidate)
                selected.append((quantile, candidate))
                break
    return selected


def interpolate_palette(values: np.ndarray, anchors: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    output = np.zeros(clipped.shape + (3,), dtype=np.float64)
    for left, right in zip(anchors[:-1], anchors[1:]):
        lo, color_lo = left
        hi, color_hi = right
        active = np.logical_and(clipped >= lo, clipped <= hi)
        weight = np.zeros_like(clipped)
        if hi > lo:
            weight[active] = (clipped[active] - lo) / (hi - lo)
        for channel in range(3):
            output[..., channel][active] = (
                color_lo[channel] * (1.0 - weight[active])
                + color_hi[channel] * weight[active]
            )
    return np.rint(output).astype(np.uint8)


VIRIDIS = [
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
]

MAGMA = [
    (0.00, (0, 0, 4)),
    (0.25, (81, 18, 124)),
    (0.50, (183, 55, 121)),
    (0.75, (252, 137, 97)),
    (1.00, (252, 253, 191)),
]


def field_image(field: np.ndarray, size: int) -> Image.Image:
    rgb = interpolate_palette((np.clip(field, -1.0, 1.0) + 1.0) / 2.0, VIRIDIS)
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def observed_image(field: np.ndarray, mask: np.ndarray, size: int) -> Image.Image:
    rgb = np.full(field.shape + (3,), 239, dtype=np.uint8)
    colored = interpolate_palette((np.clip(field, -1.0, 1.0) + 1.0) / 2.0, VIRIDIS)
    rgb[mask] = colored[mask]
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def error_image(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    error_max: float,
    size: int,
) -> Image.Image:
    error = np.abs(prediction - target)
    normalized = np.clip(error / max(error_max, 1e-12), 0.0, 1.0)
    rgb = np.full(error.shape + (3,), 242, dtype=np.uint8)
    colored = interpolate_palette(normalized, MAGMA)
    rgb[mask] = colored[mask]
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size, index=1 if bold else 0)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (24, 38, 54),
) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    width = box[2] - box[0]
    draw.text((xy[0] - width // 2, xy[1]), text, font=text_font, fill=fill)


def draw_colorbar(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    palette: list[tuple[float, tuple[int, int, int]]],
    left_label: str,
    right_label: str,
) -> None:
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :]
    gradient = np.repeat(gradient, height, axis=0)
    image = Image.fromarray(interpolate_palette(gradient, palette), mode="RGB")
    canvas.paste(image, (x, y))
    draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(71, 85, 105), width=2)
    label_font = font(21)
    draw.text((x, y + height + 8), left_label, font=label_font, fill=(51, 65, 85))
    right_box = draw.textbbox((0, 0), right_label, font=label_font)
    draw.text(
        (x + width - (right_box[2] - right_box[0]), y + height + 8),
        right_label,
        font=label_font,
        fill=(51, 65, 85),
    )


def render(arrays: dict[str, np.ndarray], output: Path) -> dict[str, object]:
    target = arrays["target"]
    train_mask = arrays["train_mask"]
    test_mask = arrays["test_mask"]
    fourier = arrays["fourier"]
    phase = arrays["phase"]

    fourier_rmse = rmse_by_field(target, fourier, test_mask)
    phase_rmse = rmse_by_field(target, phase, test_mask)
    improvement = fourier_rmse - phase_rmse
    selected = select_quantile_fields(improvement)

    selected_errors = []
    for _, index in selected:
        selected_errors.append(np.abs(fourier[index] - target[index])[test_mask[index]])
        selected_errors.append(np.abs(phase[index] - target[index])[test_mask[index]])
    error_max = float(np.quantile(np.concatenate(selected_errors), 0.98))

    panel = 250
    gap_x = 24
    gap_y = 42
    left = 176
    top = 200
    columns = 6
    rows = 3
    width = left + columns * panel + (columns - 1) * gap_x + 70
    height = top + rows * panel + (rows - 1) * gap_y + 175
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = font(38, bold=True)
    header_font = font(27, bold=True)
    row_font = font(24, bold=True)
    note_font = font(21)
    centered_text(draw, (width // 2, 38), "原始输出的场内空间重建对比", title_font)
    centered_text(
        draw,
        (width // 2, 91),
        "固定第一组随机化运行；按两种网络单场 RMSE 差的 25%、50% 和 75% 分位数选取",
        note_font,
        (71, 85, 105),
    )

    headers = [
        "真实场",
        "训练观测位置",
        "Fourier 对照",
        "周期相位网络",
        "Fourier |误差|",
        "周期相位 |误差|",
    ]
    for column, header in enumerate(headers):
        x = left + column * (panel + gap_x) + panel // 2
        centered_text(draw, (x, top - 48), header, header_font)

    metadata_rows: list[dict[str, float | int | str]] = []
    for row, (quantile, index) in enumerate(selected):
        y = top + row * (panel + gap_y)
        label = f"Q{int(quantile * 100)}  场 {index + 1}"
        draw.text((22, y + 78), label, font=row_font, fill=(30, 41, 59))
        draw.text(
            (22, y + 120),
            f"ΔRMSE={improvement[index]:.4f}",
            font=note_font,
            fill=(71, 85, 105),
        )

        panels = [
            field_image(target[index], panel),
            observed_image(target[index], train_mask[index], panel),
            field_image(fourier[index], panel),
            field_image(phase[index], panel),
            error_image(target[index], fourier[index], test_mask[index], error_max, panel),
            error_image(target[index], phase[index], test_mask[index], error_max, panel),
        ]
        for column, image in enumerate(panels):
            x = left + column * (panel + gap_x)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + panel, y + panel), outline=(51, 65, 85), width=2)

        metadata_rows.append(
            {
                "quantile": quantile,
                "sample_index_zero_based": index,
                "field_number_one_based": index + 1,
                "fourier_rmse": float(fourier_rmse[index]),
                "phase_periodic_rmse": float(phase_rmse[index]),
                "rmse_difference": float(improvement[index]),
            }
        )

    bar_y = height - 100
    draw_colorbar(canvas, draw, left, bar_y, 610, 24, VIRIDIS, "-1", "1")
    centered_text(draw, (left + 305, bar_y - 34), "归一化对数渗透率", note_font)
    error_x = left + 3 * (panel + gap_x) + 80
    draw_colorbar(canvas, draw, error_x, bar_y, 610, 24, MAGMA, "0", f"{error_max:.3f}")
    centered_text(draw, (error_x + 305, bar_y - 34), "空间留出点绝对误差（98% 分位截断）", note_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)
    return {
        "selection_rule": (
            "within the first randomized run, select the nearest fields to Q25, Q50, "
            "and Q75 of paired per-field RMSE improvement"
        ),
        "error_color_limit": "98th percentile across selected held-out absolute errors",
        "error_color_max": error_max,
        "rows": metadata_rows,
    }


def main() -> None:
    args = parse_args()
    arrays = load_arrays(args.seed_dir)
    metadata = render(arrays, args.output)
    try:
        run_directory = args.seed_dir.resolve().relative_to(ROOT)
    except ValueError:
        run_directory = args.seed_dir.resolve()
    try:
        figure_path = args.output.resolve().relative_to(ROOT)
    except ValueError:
        figure_path = args.output.resolve()
    metadata["randomized_run_directory"] = str(run_directory)
    metadata["figure"] = str(figure_path)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(sidecar)


if __name__ == "__main__":
    main()
