# V3 Darcy inverse reconstruction

This is a synthetic, dimensionless, per-field reconstruction experiment.
Exact quantum-circuit simulation is not evidence of hardware speedup or quantum advantage.
MC is a paired feature-ablation heuristic; RBA is a classical training aid.
All metrics use raw selected-checkpoint outputs, without interpolation correction.
Profile: research. Status: completed.

| Method | Runs | Test k relative L2 | Test pressure RMSE | PDE residual RMSE | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical_plain | 9 | 0.10571 | 0.00825939 | 0.0938512 | 0.832911 |
| classical_rba | 9 | 0.106315 | 0.00858368 | 0.0850648 | 0.875924 |
| classical_mc | 9 | 0.106312 | 0.00788754 | 0.0958089 | 1.04535 |
| classical_rba_mc | 9 | 0.106469 | 0.00849988 | 0.0848508 | 1.04294 |
| quantum_plain | 9 | 0.102443 | 0.010563 | 0.123228 | 6.64878 |
| quantum_rba | 9 | 0.103364 | 0.0109439 | 0.11581 | 6.62968 |
| quantum_mc | 9 | 0.101942 | 0.0104103 | 0.124264 | 9.02051 |
| quantum_rba_mc | 9 | 0.103363 | 0.0108998 | 0.115033 | 8.93359 |

See summary.json for paired differences and field-level averages.
A smoke run checks implementation, not convergence, significance, or publication readiness.
Equal epochs and shared observations are controlled; feature-block parameter counts differ.
MC screening overhead is included in training time. There is no matched-wall-time experiment here.
The pressure-network boundary conditions are hard Dirichlet and soft no-flux.
Dataset metadata reports reference grid refinement error; no field-wise normalization is used.
