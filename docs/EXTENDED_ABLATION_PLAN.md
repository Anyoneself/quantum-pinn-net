# Extended Ablation Plan

## Purpose

The previous ablation suite mainly followed one incremental path:

`Baseline -> Quantum Feature -> Quantum + Loss -> Quantum + Loss + MC -> Quantum + Loss + RBA -> Full`

The extended suite adds single-module and cross-module variants so the paper can distinguish:

- whether residual-adaptive weighting helps without quantum features;
- whether dynamic loss balancing helps the classical PINN;
- whether MC neuron thresholding helps when used without dynamic loss balancing;
- whether RBA interacts positively with quantum features independent of the loss balancer;
- whether the full model improves prediction error, physics residual, or both.

## Extended ablation variants

| Key | Model | Quantum features | Dynamic loss | MC threshold | Residual adaptive |
| --- | --- | --- | --- | --- | --- |
| `standard-pinn` | Baseline PINN | no | no | no | no |
| `standard-loss` | Classical + Loss Balancer | no | yes | no | no |
| `standard-rba` | Classical + RBA | no | no | no | yes |
| `standard-loss-rba` | Classical + Loss + RBA | no | yes | no | yes |
| `quantum-feature` | Quantum Feature PINN | yes | no | no | no |
| `quantum-mc-only` | Quantum + MC | yes | no | yes | no |
| `quantum-rba` | Quantum + RBA | yes | no | no | yes |
| `quantum-loss` | Quantum + Loss Balancer | yes | yes | no | no |
| `quantum-mc` | Quantum + Loss + MC | yes | yes | yes | no |
| `rba-no-mc` | Quantum + Loss + RBA | yes | yes | no | yes |
| `rba-qpinn` | Full RBA-QPINN | yes | yes | yes | yes |

## Run command

```bash
python3 gen-figures/run_pinn_experiment.py \
  --epochs 60 \
  --batch-size 8 \
  --hidden-dim 128 \
  --latent-dim 24 \
  --mc-trials 4 \
  --residual-resample-period 5 \
  --residual-replace-ratio 0.20 \
  --run-ablations
```

After training, regenerate the extended paper figures:

```bash
python3 gen-figures/generate_extended_paper_figures.py
```

## Figures generated

- `figures/ablation_design_matrix.png`
- `figures/ablation_study_summary.png`
- `figures/ablation_module_contribution.png`
- `figures/ablation_path_metrics.png`
- `figures/ablation_metric_heatmap.png`
- `figures/ablation_sample_rmse_heatmap.png`
- `figures/ablation_table.png`
- `figures/error_boxplot.png`
- `figures/prediction_field_ablation_comparison.png`

