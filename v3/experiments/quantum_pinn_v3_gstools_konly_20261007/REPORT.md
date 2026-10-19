# V3 Darcy inverse reconstruction

This is a synthetic, dimensionless, per-field reconstruction experiment.
Exact quantum-circuit simulation is not evidence of hardware speedup or quantum advantage.
MC is a paired feature-ablation heuristic; RBA is a classical training aid.
All metrics use raw selected-checkpoint outputs, without interpolation correction.
Profile: research. Status: completed.
Field generator: gstools. Pressure supervision: False.
Pressure reference values are evaluation-only when pressure supervision is false.

| Method | Runs | Test k relative L2 | Test pressure RMSE | PDE residual RMSE | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical_plain | 9 | 0.337094 | 0.0251624 | 0.253345 | 0.806676 |
| classical_rba | 9 | 0.338305 | 0.0256518 | 0.205412 | 0.829655 |
| classical_mc | 9 | 0.336532 | 0.0259789 | 0.256594 | 0.994967 |
| classical_rba_mc | 9 | 0.337718 | 0.0261824 | 0.201774 | 0.995277 |
| quantum_plain | 9 | 0.306759 | 0.0223943 | 0.260069 | 6.32414 |
| quantum_rba | 9 | 0.305936 | 0.0219798 | 0.197464 | 6.35114 |
| quantum_mc | 9 | 0.306767 | 0.0234391 | 0.246591 | 8.85453 |
| quantum_rba_mc | 9 | 0.305882 | 0.0212302 | 0.209573 | 8.77151 |

See summary.json for paired differences and field-level averages.
A smoke run checks implementation, not convergence, significance, or publication readiness.
Equal epochs and shared observations are controlled; feature-block parameter counts differ.
MC screening overhead is included in training time. There is no matched-wall-time experiment here.
The pressure-network boundary conditions are hard Dirichlet and soft no-flux.
Dataset metadata reports reference grid refinement error; no field-wise normalization is used.
Darcy consistency without pressure observations does not uniquely identify permeability.
GSTools runs generate new fields; they do not reuse V2's normalized/clipped arrays.
