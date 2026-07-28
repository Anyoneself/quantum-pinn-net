# RBA-QPINN submission experiment v1

## Protocol

- Seeds: 42, 43, 44, 45, 46
- Spatial split: 70% train, 10% validation, 20% independent test
- Models per seed: 6
- Epochs: 40
- Primary result: Raw network output
- Secondary result: IDW-corrected output using training-point residuals only
- Output protection: `--fail-if-output-exists`

## Reproduction

Run once for each seed, replacing `<seed>`:

```bash
python3 gen-figures/run_pinn_experiment.py \
  --output-dir experiments/rba_qpinn_submission_v1/seed_<seed> \
  --processed-dir experiments/rba_qpinn_submission_v1/seed_<seed>/processed \
  --fail-if-output-exists \
  --paper-ablations \
  --seed <seed> \
  --epochs 40 \
  --batch-size 8 \
  --hidden-dim 128 \
  --depth 4 \
  --latent-dim 24 \
  --learning-rate 0.001 \
  --train-ratio 0.70 \
  --validation-ratio 0.10 \
  --max-samples 128 \
  --device cpu \
  --mc-trials 4 \
  --residual-resample-period 5 \
  --residual-replace-ratio 0.20 \
  --no-darcy-physics
```

Aggregate all five seeds:

```bash
python3 gen-figures/aggregate_multiseed_results.py \
  --experiment-root experiments/rba_qpinn_submission_v1 \
  --seeds 42,43,44,45,46
```

## Main outputs

- `aggregate/aggregate_metrics_raw.csv`
- `aggregate/aggregate_metrics_corrected.csv`
- `aggregate/per_seed_metrics.csv`
- `aggregate/statistical_tests.json`
- `aggregate/multiseed_rmse_raw_vs_corrected.png`
- `aggregate/environment.json`

Each seed directory contains the exact configuration, masks, model weights, Raw/Corrected predictions, aggregate metrics, and per-sample metrics.
