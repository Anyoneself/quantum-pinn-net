from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/quantum-pinn-v2-mpl")
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import scipy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "v2/src/experiment"))
import run_pinn_experiment as core

WEIGHTS = {"data": 1.0, "physics": 0.1, "smoothness": 0.002, "boundary": 0.001}
METHODS = [
    "fourier", "phase", "no-mixer", "no-phase", "no-sine", "no-gate",
    "no-condition", "data-only", "rba", "dynamic", "full",
]
SCENARIOS = {
    "random70": (0.7, 0.0, False),
    "random40": (0.4, 0.0, False),
    "random20": (0.2, 0.0, False),
    "noise05": (0.7, 0.05, False),
    "noise10": (0.7, 0.10, False),
    "block": (0.7, 0.0, True),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masks(shape: tuple[int, ...], rate: float, seed: int, block: bool = False):
    rng = np.random.default_rng(seed + 2027)
    n, h, w = shape
    train = np.zeros(shape, bool)
    val = np.zeros(shape, bool)
    test = np.zeros(shape, bool)
    for s in range(n):
        if block:
            test[s, h // 2 - 7:h // 2 + 7, w // 2 - 7:w // 2 + 7] = True
            order = rng.permutation(np.flatnonzero(~test[s].ravel()))
            val[s].ravel()[order[:int(h * w * 0.1)]] = True
            train[s].ravel()[order[int(h * w * 0.1):int(h * w * 0.1) + int(h * w * rate)]] = True
        else:
            order = rng.permutation(h * w)
            # Fixed held-out sets make observation-rate comparisons paired.
            ntest = h * w - int(h * w * 0.7) - int(h * w * 0.1)
            nval = int(h * w * 0.1)
            test[s].ravel()[order[:ntest]] = True
            val[s].ravel()[order[ntest:ntest + nval]] = True
            train[s].ravel()[order[ntest + nval:ntest + nval + int(h * w * rate)]] = True
    return train, val, test


def observed_values(truth: np.ndarray, train: np.ndarray, val: np.ndarray,
                    noise: float, seed: int):
    rng = np.random.default_rng(seed + 809)
    observed = np.zeros_like(truth)
    for s in range(len(truth)):
        available = train[s] | val[s]
        scale = float(np.std(truth[s][train[s]], ddof=0))
        observed[s][available] = truth[s][available] + rng.normal(
            0, noise * scale, int(available.sum())
        )
    return observed


def fit_prior(train_values: np.ndarray, train: np.ndarray, conditions: np.ndarray):
    means = np.array([train_values[s][train[s]].mean() for s in range(len(train))])
    index = core.expected_log_perm_index(conditions, core.PhysicalConfig()).astype(float)
    design = np.column_stack([index, np.ones(len(index))])
    coef = np.linalg.lstsq(design, means, rcond=None)[0]
    return (design @ coef).astype(np.float32), {"slope": float(coef[0]), "intercept": float(coef[1])}


class PhaseAblation(core.QuantumFeaturePINN):
    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        if mode == "no-mixer":
            self.quantum_mixer.requires_grad_(False)
        if mode == "no-phase":
            self.quantum_mixer.requires_grad_(False)
            self.angle_encoder.requires_grad_(False)
        if mode == "no-gate":
            self.gate.requires_grad_(False)

    def quantum_features(self, x):
        if self.mode == "no-phase":
            return x.new_zeros((len(x), self.qubits * 4))
        if self.mode == "no-mixer":
            a = self.angle_encoder(x)
            return torch.cat([a.sin(), a.cos(), a.sin(), a.cos()], -1)
        return super().quantum_features(x)

    def forward(self, x):
        if self.mode == "no-condition":
            x = torch.cat([x[:, :2], torch.zeros_like(x[:, 2:7]), x[:, 7:]], -1)
        features = torch.cat([x, self.quantum_features(x)], -1)
        first = (F.silu(self.first_layer.linear(features)) if self.mode == "no-sine"
                 else self.first_layer(features))
        hidden = first if self.mode == "no-gate" else torch.sin(self.gate(features)) * first
        for layer in self.hidden_layers:
            hidden = hidden + F.silu(layer(hidden))
        return self.output(hidden)


def make_model(method: str, count: int):
    kwargs = dict(input_dim=31, hidden_dim=128, depth=4, sample_count=count, latent_dim=24)
    return core.StandardPINN(**kwargs) if method == "fourier" else PhaseAblation(method, **kwargs)


def metrics(truth: np.ndarray, pred: np.ndarray, test: np.ndarray):
    diff = (pred.astype(float) - truth.astype(float))[test]
    y = truth.astype(float)[test]
    rmse = np.sqrt(np.mean(diff ** 2))
    per_field = np.array([
        np.sqrt(np.mean((pred[s].astype(float)[test[s]] - truth[s][test[s]]) ** 2))
        for s in range(len(truth))
    ])
    gx = np.gradient(truth.astype(float), axis=2)
    gy = np.gradient(truth.astype(float), axis=1)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    cutoff = np.quantile(gradient[test], 0.75)
    high = test & (gradient >= cutoff)
    return {
        "rmse": float(rmse), "mae": float(np.mean(abs(diff))),
        "r2": float(1 - np.sum(diff ** 2) / np.sum((y - y.mean()) ** 2)),
        "field_mean_rmse": float(np.sqrt(np.mean((pred.mean((1, 2)) - truth.mean((1, 2))) ** 2))),
        "high_gradient_rmse": float(np.sqrt(np.mean((pred[high] - truth[high]) ** 2))),
        "per_field_rmse": per_field.tolist(), "test_points": int(test.sum()),
    }


def idw(values: np.ndarray, train: np.ndarray, k: int = 8, power: float = 2.0):
    n, h, w = values.shape
    coords = np.indices((h, w)).reshape(2, -1).T
    pred = np.empty_like(values)
    for s in range(n):
        idx = np.flatnonzero(train[s].ravel())
        d, neighbors = cKDTree(coords[idx]).query(coords, k=min(k, len(idx)))
        if d.ndim == 1:
            d, neighbors = d[:, None], neighbors[:, None]
        weights = 1.0 / (d ** power + 1e-6)
        pred[s] = ((weights * values[s].ravel()[idx][neighbors]).sum(1) / weights.sum(1)).reshape(h, w)
        pred[s][train[s]] = values[s][train[s]]
    return pred


def correct(pred: np.ndarray, train_values: np.ndarray, train: np.ndarray):
    residual = np.where(train, train_values - pred, 0)
    output = pred + 0.95 * idw(residual, train)
    return np.where(train, train_values, output).astype(np.float32)


def interpolate(kind: str, values: np.ndarray, train: np.ndarray, **params):
    if kind == "idw":
        return idw(values, train, **params)
    n, h, w = values.shape
    coords = np.indices((h, w)).reshape(2, -1).T / (h - 1)
    result = np.empty_like(values)
    for s in range(n):
        m = train[s].ravel()
        result[s] = RBFInterpolator(
            coords[m], values[s].ravel()[m], kernel="thin_plate_spline", **params
        )(coords).reshape(h, w)
    return result


def run_interpolator(kind, observed, train, val, out):
    choices = ([{"k": k, "power": p} for k in (4, 8, 16) for p in (1, 2, 3)]
               if kind == "idw" else
               [{"neighbors": k, "smoothing": p} for k in (16, 32, 64) for p in (0, 0.01, 0.1)])
    t0 = time.perf_counter()
    scores = []
    best = None
    best_mse = float("inf")
    for params in choices:
        pred = interpolate(kind, observed, train, **params)
        mse = float(np.mean((pred[val] - observed[val]) ** 2))
        scores.append({"parameters": params, "validation_mse": mse})
        if mse < best_mse:
            best_mse, best = mse, pred
    elapsed = time.perf_counter() - t0
    selected = min(scores, key=lambda x: x["validation_mse"])["parameters"]
    t0 = time.perf_counter()
    best = interpolate(kind, observed, train, **selected)
    inference = time.perf_counter() - t0
    write_json(out / "history.json", {"parameter_search": scores, "selected": selected})
    return best, {"training_seconds": elapsed, "inference_seconds": inference,
                  "parameters": 0, "trainable_parameters": 0, "selected": selected}


def run_network(method, observed, conditions, train, val, seed, epochs, out):
    core.set_seed(seed)
    model = make_model(method, len(observed))
    prior, prior_fit = fit_prior(observed, train, conditions)
    cn = ((conditions - conditions.mean(0)) / (conditions.std(0) + 1e-8)).astype(np.float32)
    # Neither the training nor the validation dataset contains test targets.
    dataset = core.FieldDataset(np.where(train, observed, 0), cn, conditions, prior,
                                train, val, np.zeros_like(train))
    validation = core.FieldDataset(np.where(val, observed, 0), cn, conditions, prior,
                                   train, val, np.zeros_like(train))
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 17))
    val_loader = DataLoader(validation, batch_size=8, shuffle=False)
    mc_loader = DataLoader(dataset, batch_size=8, shuffle=True,
                           generator=torch.Generator().manual_seed(seed + 913))
    coord = core.make_coordinate_grid(observed.shape[-1], torch.device("cpu"))
    config = core.Config(epochs=epochs, darcy_physics=False, hidden_dim=128, latent_dim=24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    balancer = core.QuantumLossBalancer(list(WEIGHTS)) if method in ("dynamic", "full") else None
    adaptive = core.ResidualAdaptiveWeightMap(train, config) if method in ("rba", "full") else None
    weights = {"data": 1.0, "physics": 0.0, "smoothness": 0.0, "boundary": 0.0} if method == "data-only" else WEIGHTS
    history = []
    best_mse, best_state, best_epoch = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {k: 0.0 for k in WEIGHTS}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses, _ = core.compute_losses(model, batch, coord, "train_mask",
                                           use_point_weights=adaptive is not None, config=config)
            used = balancer.weights(losses) if balancer else weights
            total = sum(used[k] * losses[k] for k in WEIGHTS)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last = {k: float(losses[k].detach()) for k in WEIGHTS}
            for k in WEIGHTS:
                sums[k] += last[k] / len(train_loader)
        used_float = {k: float(v) for k, v in used.items()}
        scheduler.step()
        if balancer:
            balancer.update(last)
        reset = 0
        if method == "full" and epoch <= epochs // 2:
            reset = core.monte_carlo_neuron_screen(model, mc_loader, coord, 4)
        pred = core.predict_dataset_fields(model, val_loader, coord)
        mse = float(np.mean((pred[val] - observed[val]) ** 2))
        if adaptive:
            adaptive.update_from_prediction(pred, dataset.fields.numpy(), epoch)
            dataset.update_point_weight(adaptive.snapshot())
        history.append({"epoch": epoch, "train": sums, "weights_used": used_float,
                        "validation_mse": mse, "mc_reset": reset})
        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch in (1, epochs) or epoch % 10 == 0:
            print(f"  {method} epoch {epoch}/{epochs} val_rmse={mse ** 0.5:.6f}", flush=True)
    training = time.perf_counter() - start
    model.load_state_dict(best_state)
    start = time.perf_counter()
    pred = core.predict_dataset_fields(model, val_loader, coord)
    inference = time.perf_counter() - start
    torch.save(best_state, out / "model.pt")
    write_json(out / "history.json", history)
    return pred, {
        "training_seconds": training, "inference_seconds": inference,
        "best_epoch": best_epoch, "validation_mse": best_mse, "prior_fit": prior_fit,
        "parameters": sum(x.numel() for x in model.parameters()),
        "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
    }


def flow_response(field: np.ndarray):
    """Cell-centred finite volumes, unit domain, two Dirichlet boundary faces."""
    h, w = field.shape
    k = np.exp(field.astype(float) - float(field.mean()))
    idx = np.arange(h * w).reshape(h, w)
    gx = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:]) * w / h
    gy = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:]) * h / w
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:].ravel()])
    g = np.concatenate([gx.ravel(), gy.ravel()])
    diagonal = np.bincount(np.concatenate([a, b]), weights=np.tile(g, 2), minlength=h * w)
    left, right = idx[:, 0], idx[:, -1]
    gl, gr = 2 * k[:, 0] * w / h, 2 * k[:, -1] * w / h
    diagonal[left] += gl
    diagonal[right] += gr
    rhs = np.zeros(h * w)
    rhs[left] = gl
    matrix = coo_matrix(
        (np.concatenate([-g, -g, diagonal]),
         (np.concatenate([a, b, idx.ravel()]), np.concatenate([b, a, idx.ravel()]))),
        shape=(h * w, h * w),
    ).tocsr()
    p = spsolve(matrix, rhs)
    inlet = float(np.sum(gl * (1 - p[left])))
    outlet = float(np.sum(gr * p[right]))
    residual = float(np.max(abs(matrix @ p - rhs)))
    return p.reshape(h, w), inlet, abs(inlet - outlet) / max(abs(inlet), 1e-15), residual


def flow_metrics(truth: np.ndarray, pred: np.ndarray):
    pressure, flux, residuals, imbalance = [], [], [], []
    for actual, estimate in zip(truth, pred):
        p0, q0, _, _ = flow_response(actual)
        p1, q1, balance, residual = flow_response(estimate)
        pressure.append(float(np.sqrt(np.mean((p1 - p0) ** 2))))
        flux.append(abs(q1 - q0) / abs(q0))
        residuals.append(residual)
        imbalance.append(balance)
    return {"pressure_rmse_mean": float(np.mean(pressure)),
            "flux_relative_error_mean": float(np.mean(flux)),
            "pressure_per_field": pressure, "flux_per_field": flux,
            "max_solver_residual": float(max(residuals)),
            "max_flux_imbalance": float(max(imbalance))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="v2/experiments/supplement_v2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-fields", type=int, default=0)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    data = ROOT / "v1/experiments/rba_qpinn_submission_v1/seed_42/processed"
    truth = np.load(data / "fields.npy")
    conditions = np.load(data / "conditions.npy")
    if args.max_fields:
        truth, conditions = truth[:args.max_fields], conditions[:args.max_fields]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = {
        "protocol": "v2/paper/materical/SUPPLEMENT_EXPERIMENT_PROTOCOL_V2.md",
        "data_sha256": digest(data / "fields.npy"), "condition_sha256": digest(data / "conditions.npy"),
        "script_sha256": digest(Path(__file__)), "arguments": vars(args),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "field_count": len(truth), "device": "cpu", "shape": list(truth.shape),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise RuntimeError("Existing manifest differs. Use a new output directory.")
    else:
        write_json(manifest_path, manifest)
    for seed in args.seeds:
        for scenario, (rate, noise, block) in SCENARIOS.items():
            if scenario not in args.scenarios:
                continue
            train, val, test = masks(truth.shape, rate, seed, block)
            observed = observed_values(truth, train, val, noise, seed)
            directory = output / f"seed_{seed}" / scenario
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / "observations.npz", train=train, validation=val,
                                test=test, observed=observed)
            methods = args.methods or (["idw", "rbf"] + (METHODS if scenario == "random70" else ["fourier", "phase"]))
            for method in methods:
                out = directory / method
                if (out / "result.json").exists():
                    print(f"SKIP completed {seed} {scenario} {method}", flush=True)
                    continue
                out.mkdir(parents=True, exist_ok=True)
                print(f"RUN {seed} {scenario} {method}", flush=True)
                if method in ("idw", "rbf"):
                    pred, info = run_interpolator(method, observed, train, val, out)
                else:
                    pred, info = run_network(method, observed, conditions, train, val, seed, args.epochs, out)
                result = {"seed": seed, "scenario": scenario, "method": method, "epochs": args.epochs,
                          **info, "raw": metrics(truth, pred, test)}
                np.save(out / "raw.npy", pred)
                if method not in ("idw", "rbf"):
                    t0 = time.perf_counter()
                    corrected = correct(pred, observed, train)
                    result["correction_seconds"] = time.perf_counter() - t0
                    result["corrected"] = metrics(truth, corrected, test)
                    np.save(out / "corrected.npy", corrected)
                if method in ("idw", "rbf", "fourier", "phase") and scenario in ("random70", "block"):
                    result["flow_raw"] = flow_metrics(truth, pred)
                    if method not in ("idw", "rbf"):
                        result["flow_corrected"] = flow_metrics(truth, corrected)
                write_json(out / "result.json", result)
                print(f"DONE {seed} {scenario} {method} rmse={result['raw']['rmse']:.6f} "
                      f"seconds={info['training_seconds']:.1f}", flush=True)
    if args.extended:
        seed = args.seeds[0]
        train, val, test = masks(truth.shape, 0.7, seed)
        observed = observed_values(truth, train, val, 0, seed)
        for method in ("fourier", "phase"):
            out = output / f"seed_{seed}" / "budget120" / method
            if (out / "result.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            pred, info = run_network(method, observed, conditions, train, val, seed, 120, out)
            np.save(out / "raw.npy", pred)
            write_json(out / "result.json", {"seed": seed, "scenario": "budget120", "method": method,
                       "epochs": 120, **info, "raw": metrics(truth, pred, test)})
    print("ALL REQUESTED RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
