# V3 Darcy inverse reconstruction

This is a synthetic, dimensionless, per-field reconstruction experiment.
Exact quantum-circuit simulation is not evidence of hardware speedup or quantum advantage.
MC is a paired feature-ablation heuristic; RBA is a classical training aid.
All metrics use raw selected-checkpoint outputs, without interpolation correction.
Profile: smoke. Status: completed.
Field generator: gstools. Pressure supervision: False.
Pressure reference values are evaluation-only when pressure supervision is false.

| Method | Runs | Test k relative L2 | Test pressure RMSE | PDE residual RMSE | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical_plain | 1 | 0.447773 | 0.0598678 | 0.444635 | 0.071373 |
| classical_rba | 1 | 0.447773 | 0.0598678 | 0.444635 | 0.078157 |
| classical_mc | 1 | 0.447773 | 0.0598678 | 0.444635 | 0.091514 |
| classical_rba_mc | 1 | 0.447773 | 0.0598678 | 0.444635 | 0.092112 |
| quantum_plain | 1 | 0.446078 | 0.0598678 | 0.455861 | 0.602896 |
| quantum_rba | 1 | 0.446078 | 0.0598678 | 0.455861 | 0.63391 |
| quantum_mc | 1 | 0.446078 | 0.0598678 | 0.455861 | 0.821447 |
| quantum_rba_mc | 1 | 0.446078 | 0.0598678 | 0.455861 | 0.80586 |

See summary.json for paired differences and field-level averages.
A smoke run checks implementation, not convergence, significance, or publication readiness.
Equal epochs and shared observations are controlled; feature-block parameter counts differ.
MC screening overhead is included in training time. There is no matched-wall-time experiment here.
The pressure-network boundary conditions are hard Dirichlet and soft no-flux.
Dataset metadata reports reference grid refinement error; no field-wise normalization is used.
Darcy consistency without pressure observations does not uniquely identify permeability.
GSTools runs generate new fields; they do not reuse V2's normalized/clipped arrays.
