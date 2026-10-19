import argparse
import json
from pathlib import Path

from .data import DataConfig
from .experiment import run_experiment, verify_experiment
from .models import ModelConfig
from .physics import LossWeights
from .strategies import StrategyConfig
from .training import METHODS, TrainConfig


def parser():
    root = argparse.ArgumentParser(description="V3 quantum-feature Darcy PINN experiments")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a controlled matrix in a NEW output directory")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--smoke", action="store_true", help="small implementation check, not research evidence")
    run.add_argument("--field-seeds", type=int, nargs="+")
    run.add_argument("--seeds", type=int, nargs="+")
    run.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    run.add_argument("--epochs", type=int)
    run.add_argument("--grid", type=int)
    run.add_argument("--refinement", type=int, default=3)
    run.add_argument("--modes", type=int, default=4)
    run.add_argument("--log-amplitude", type=float, default=0.6)
    run.add_argument("--k-train", type=int)
    run.add_argument("--p-train", type=int)
    run.add_argument("--validation-points", type=int)
    run.add_argument("--collocation", type=int)
    run.add_argument("--boundary", type=int)
    run.add_argument("--noise", type=float, default=0.0)
    run.add_argument("--qubits", type=int, default=4, help="also the classical feature count")
    run.add_argument("--feature-layers", type=int, default=2)
    run.add_argument("--hidden", type=int, default=24)
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--warmup", type=int)
    run.add_argument("--mc-every", type=int)
    run.add_argument("--mc-trials", type=int)
    run.add_argument("--mc-batch", type=int)
    run.add_argument("--mc-threshold", type=float, default=0.01)
    run.add_argument("--mc-fraction", type=float, default=0.25)
    run.add_argument("--min-features", type=int, default=2)
    run.add_argument("--rba-decay", type=float, default=0.9)
    run.add_argument("--rba-gain", type=float, default=4.0)
    check = commands.add_parser("verify", help="read-only verification of a completed archive")
    check.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify_experiment(args.output), indent=2))
        return

    def setting(name, smoke, research):
        value = getattr(args, name)
        return value if value is not None else (smoke if args.smoke else research)

    validation = setting("validation_points", 8, 24)
    data = DataConfig(
        grid=setting("grid", 12, 24), refinement=args.refinement, modes=args.modes,
        log_amplitude=args.log_amplitude, k_train=setting("k_train", 16, 48),
        k_validation=validation, p_train=setting("p_train", 16, 48), p_validation=validation,
        collocation=setting("collocation", 32, 128), boundary=setting("boundary", 8, 32),
        noise=args.noise,
    )
    model = ModelConfig(features=args.qubits, feature_layers=args.feature_layers, hidden=args.hidden)
    train = TrainConfig(epochs=setting("epochs", 30, 300), learning_rate=args.learning_rate, threads=args.threads)
    strategy = StrategyConfig(
        warmup=setting("warmup", 5, 20), mc_every=setting("mc_every", 10, 20),
        mc_trials=setting("mc_trials", 2, 4), mc_batch=setting("mc_batch", 8, 32),
        mc_threshold=args.mc_threshold, mc_fraction=args.mc_fraction, min_features=args.min_features,
        rba_decay=args.rba_decay, rba_gain=args.rba_gain,
    )
    run_experiment(args.output, data, model, train, strategy, LossWeights(),
                   setting("field_seeds", [11], [11, 22, 33]),
                   setting("seeds", [101], [101, 202, 303]), args.methods,
                   profile="smoke" if args.smoke else "research")
    print(json.dumps(verify_experiment(args.output), indent=2))


if __name__ == "__main__":
    main()
