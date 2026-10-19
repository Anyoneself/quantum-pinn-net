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
| classical_plain | 3 | 0.249121 | 0.0187645 | 0.277037 | 13.1232 |
| quantum_plain | 3 | 0.232923 | 0.0164982 | 0.197958 | 108.904 |

See summary.json for paired differences and field-level averages.
A smoke run checks implementation, not convergence, significance, or publication readiness.
Equal epochs and shared observations are controlled; feature-block parameter counts differ.
MC screening overhead is included in training time. There is no matched-wall-time experiment here.
The pressure-network boundary conditions are hard Dirichlet and soft no-flux.
Dataset metadata reports reference grid refinement error; no field-wise normalization is used.
Darcy consistency without pressure observations does not uniquely identify permeability.
GSTools runs generate new fields; they do not reuse V2's normalized/clipped arrays.
