"""有限宽度经验NTK：复用V3计算图，不使用无限宽度或固定核假设。"""

import numpy as np
import torch

from metrics.physics import components, tensor


GROUPS = ("encoder", "features", "readout", "pressure")


def parameter_group(name):
    """按已有模块边界划分雅可比列，避免把压力网络收益归给量子特征。"""
    return "pressure" if name.startswith("pressure.") else name.split(".")[1]


def empirical_jacobian(values, named_parameters):
    """逐标量反传兼容PennyLane；只在得到参数导数后断开计算图。"""
    named = [(name, p) for name, p in named_parameters if p.requires_grad]
    if not named or values.numel() == 0 or not values.requires_grad:
        raise ValueError("NTK requires nonempty differentiable values and trainable parameters")
    rows = []
    for value in values.reshape(-1):
        gradients = torch.autograd.grad(
            value, tuple(p for _, p in named), retain_graph=True, allow_unused=True)
        row = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                         for (_, p), g in zip(named, gradients)])
        rows.append(row.detach())
    jacobian = torch.stack(rows).cpu().numpy()
    if not np.isfinite(jacobian).all():
        raise FloatingPointError("non-finite parameter Jacobian")
    columns = {}
    offset = 0
    for name, p in named:
        group = parameter_group(name)
        columns.setdefault(group, []).extend(range(offset, offset + p.numel()))
        offset += p.numel()
    return jacobian, columns


def weighted_errors(model, training, weights):
    """平方范数严格等于无RBA的原损失；默认压力标签为空，不访问参考压力。"""
    errors = components(model, training)
    pieces, slices = [], {}
    offset = 0
    for name, error in errors.items():
        weight = getattr(weights, name)
        if error.numel() and weight > 0:
            pieces.append((weight / error.numel())**0.5 * error)
            slices[name] = [offset, offset + error.numel()]
            offset += error.numel()
    return torch.cat(pieces), slices


def kernel_statistics(kernel, initial, direction):
    """原始尺度与迹归一化形状分别报告；退化核不伪造条件数或对齐分数。"""
    kernel = (kernel + kernel.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(kernel)[::-1]
    scale = max(float(eigenvalues[0]), np.finfo(float).tiny)
    if eigenvalues[-1] < -1e-9 * scale:
        raise ValueError("kernel is not positive semidefinite within relative tolerance")
    nonnegative = np.maximum(eigenvalues, 0)
    trace = float(np.trace(kernel))
    initial_trace = float(np.trace(initial))
    if min(trace, initial_trace) <= 0:
        raise ValueError("zero-trace kernel cannot be normalized")
    normalized = kernel / trace
    normalized_initial = initial / initial_trace
    probabilities = nonnegative / nonnegative.sum()
    positive = probabilities[probabilities > 0]
    resolved = nonnegative[nonnegative > scale * 1e-10]
    norm = float(np.linalg.norm(direction))
    alignment = (float(direction @ kernel @ direction / (trace * norm**2))
                 if norm > 1e-14 else None)
    stats = {
        "trace": trace,
        "trace_ratio_to_initial": trace / initial_trace,
        "raw_drift": float(np.linalg.norm(kernel - initial) / np.linalg.norm(initial)),
        "shape_drift": float(np.linalg.norm(normalized - normalized_initial)
                             / np.linalg.norm(normalized_initial)),
        "effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "resolved_rank_rtol_1e_10": int(len(resolved)),
        "resolved_condition": float(resolved[0] / resolved[-1]) if len(resolved) else None,
        "direction_alignment": alignment,
        "min_eigenvalue_raw": float(eigenvalues[-1]),
    }
    return stats, eigenvalues


def cosine_probes(xy):
    """固定余弦探针不是核特征向量，也不要求渗透率满足周期边界条件。"""
    modes = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2),
             (2, 2), (3, 0), (0, 3), (3, 3))
    result = {}
    for nx, ny in modes:
        phi = np.cos(np.pi * nx * xy[:, 0]) * np.cos(np.pi * ny * xy[:, 1])
        phi = phi - phi.mean()
        result[f"cos_{nx}_{ny}"] = phi / np.linalg.norm(phi)
    return result


def analyze_checkpoint(model, training, weights, probe_xy, probe_truth, initial=None):
    """保存完整雅可比、核及分组贡献；隐藏真值仅用于事后诊断，不进入训练或选模。"""
    initial = {} if initial is None else initial
    named = list(model.named_parameters())
    outputs = model.permeability(tensor(probe_xy))
    loss_errors, slices = weighted_errors(model, training, weights)
    target_centered = probe_truth - np.mean(probe_truth)
    values = {"output": outputs, "loss": loss_errors}
    directions = {"output": target_centered, "loss": loss_errors.detach().cpu().numpy()}
    arrays = {"probe_xy": probe_xy, "probe_truth": probe_truth,
              "probe_prediction": outputs.detach().cpu().numpy(),
              "weighted_training_errors": directions["loss"]}
    statistics = {}
    columns = None
    for name, value in values.items():
        jacobian, columns = empirical_jacobian(value, named)
        kernel = jacobian @ jacobian.T
        stats, eigenvalues = kernel_statistics(
            kernel, initial.get(f"{name}_kernel", kernel), directions[name])
        arrays[f"{name}_jacobian"] = jacobian
        arrays[f"{name}_kernel"] = kernel
        arrays[f"{name}_eigenvalues"] = eigenvalues
        for group in GROUPS:
            indices = columns.get(group, [])
            group_jacobian = jacobian[:, indices]
            contribution = group_jacobian @ group_jacobian.T
            arrays[f"{name}_kernel_{group}"] = contribution
            stats[f"trace_fraction_{group}"] = float(np.trace(contribution) / stats["trace"])
        statistics[name] = stats
    # 对输出核使用同一组探针，归一化响应与相应预测误差共同保存。
    centered_error = arrays["probe_prediction"] - probe_truth
    frequencies = []
    for name, phi in cosine_probes(probe_xy).items():
        kernel = arrays["output_kernel"]
        frequencies.append({
            "mode": name,
            "response_raw": float(phi @ kernel @ phi),
            "response_trace_normalized": float(phi @ kernel @ phi / np.trace(kernel)),
            "error_projection": float(phi @ centered_error),
            "error_projection_squared": float((phi @ centered_error)**2),
        })
    return arrays, statistics, frequencies, {"loss_slices": slices, "parameter_columns": columns}
