# Pretrained baseline: executed results

Model: `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
Parameters: 134,515,008. CPU float32, 3 threads.
Trial-loop wall time: 371.2 s; model loading: 3.74 s; peak process RSS: 1316.2 MiB.

No weight interventions were performed. These are public development tasks, not an independent final test.

## Metrics by group

Each sampling method has three repeats per function. Values below average repeats and functions equally. Lower errors and regret are better.

| Group | Method | Interpolation NMSE | Extrapolation NMSE | Observation-objective regret | Duplicate rate | Exact target selection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| representable | conditioned | 1.03434 | 113.645 | 0.782192 | 54.2% | 20.0% |
| representable | no_observations | 0.727407 | 49.4705 | 0.701689 | 54.2% | 26.7% |
| representable | uniform | 2.86376 | 195.761 | 2.51706 | 24.2% | 40.0% |
| representable | exhaustive | 0 | 0 | 0 | 0.0% | 100.0% |
| outside_catalog | conditioned | 15.8124 | 14.5231 | 0.96945 | 55.0% | 0.0% |
| outside_catalog | no_observations | 15.6907 | 16.3875 | 1.00371 | 47.5% | 0.0% |
| outside_catalog | uniform | 14.8881 | 36.4995 | 1.37801 | 20.8% | 0.0% |
| outside_catalog | exhaustive | 12.2468 | 35.6115 | 0 | 0.0% | 0.0% |
| all | conditioned | 8.42339 | 64.0841 | 0.875821 | 54.6% | 10.0% |
| all | no_observations | 8.20905 | 32.929 | 0.852702 | 50.8% | 13.3% |
| all | uniform | 8.87593 | 116.13 | 1.94753 | 22.5% | 20.0% |
| all | exhaustive | 6.12341 | 17.8057 | 0 | 0.0% | 50.0% |

Exact target selection is literal string recovery; it is not symbolic equivalence and is only informative for the representable group.

## Error scale and validity

| Method | Interpolation MSE | Extrapolation MSE | Numerical validity | Failed selections |
| --- | ---: | ---: | ---: | ---: |
| conditioned | 0.213013 | 2.11524 | 100.0% | 0 |
| no_observations | 0.179556 | 2.94255 | 100.0% | 0 |
| uniform | 0.221109 | 3.08364 | 100.0% | 0 |
| exhaustive | 0.0663894 | 1.16035 | 100.0% | 0 |

## Per-function errors

Cells show mean interpolation / extrapolation NMSE. Sampling methods are averaged over three seeds.

| Target | Conditioned | No observations | Uniform | Exhaustive |
| --- | ---: | ---: | ---: | ---: |
| `sin(x)` | 1.9782 / 552.46 | 1.3487 / 235.85 | 0.33516 / 341.34 | 0 / 0 |
| `cos(x)` | 0.12109 / 1.1197 | 0.12109 / 1.1197 | 5.6869 / 30.711 | 0 / 0 |
| `x*x` | 2.9579 / 10.336 | 2.0527 / 6.0777 | 1.0751 / 4.9819 | 0 / 0 |
| `exp(x)` | 0.11449 / 4.3077 | 0.11449 / 4.3077 | 2.6839 / 7.1686 | 0 / 0 |
| `1/(x*x+1)` | 0 / 0 | 0 / 0 | 4.5377 / 594.61 | 0 / 0 |
| `sin(2*x)` | 0.75403 / 11.242 | 0.82149 / 17.972 | 0.40876 / 4.5073 | 0.098605 / 9.0577 |
| `cos(2*x)` | 0.76707 / 55.566 | 0.76707 / 55.566 | 1.3942 / 63.074 | 0.76707 / 55.566 |
| `2*x*x` | 2.2605 / 0.88788 | 1.5843 / 3.48 | 1.2916 / 1.6764 | 0.8071 / 2.0707 |
| `exp(x/2)` | 0.89834 / 1.3883 | 0.89834 / 1.3883 | 1.3881 / 27.637 | 0.89834 / 1.3883 |
| `1/(x*x+2)` | 74.382 / 3.5311 | 74.382 / 3.5311 | 69.958 / 85.603 | 58.663 / 109.97 |

## Coverage diagnostics

| Target | In inventory | Minimum observation NMSE | Observation-selected catalog entry |
| --- | --- | ---: | --- |
| `sin(x)` | True | 0 | `sin(x)` |
| `cos(x)` | True | 0 | `cos(x)` |
| `x*x` | True | 0 | `x*x` |
| `exp(x)` | True | 0 | `exp(x)` |
| `1/(x*x+1)` | True | 0 | `1/(x*x+1)` |
| `sin(2*x)` | False | 0.0982586 | `x` |
| `cos(2*x)` | False | 0.938603 | `1/(x*x+1)` |
| `2*x*x` | False | 0.75011 | `x*x` |
| `exp(x/2)` | False | 0.926663 | `x+1` |
| `1/(x*x+2)` | False | 40.4896 | `sin(x*x)` |

The observation selector includes the complexity penalty. Neither the selector nor the coverage diagnostic uses reserved errors to choose an expression.

## Proposal frequencies

| Expression | Conditioned | No observations | Uniform |
| --- | ---: | ---: | ---: |
| `x` | 0 | 1 | 19 |
| `0` | 21 | 12 | 16 |
| `1` | 0 | 0 | 10 |
| `2*x` | 7 | 9 | 22 |
| `x+1` | 31 | 71 | 12 |
| `x*x` | 2 | 6 | 18 |
| `x*x+1` | 70 | 16 | 22 |
| `sin(x)` | 0 | 0 | 16 |
| `cos(x)` | 0 | 0 | 15 |
| `exp(x)` | 0 | 0 | 16 |
| `1/(x*x+1)` | 100 | 97 | 24 |
| `x/(x*x+1)` | 9 | 28 | 16 |
| `sin(x)+x` | 0 | 0 | 18 |
| `sin(x*x)` | 0 | 0 | 16 |

## Cost accounting

| Method | Candidate evaluations | All evaluator calls | Generation calls | Generated tokens | Method seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| conditioned | 240 | 300 | 30 | 1603 | 306.0870 |
| no_observations | 240 | 300 | 30 | 1591 | 64.7816 |
| uniform | 240 | 300 | 0 | 0 | 0.0400 |
| exhaustive | 140 | 160 | 0 | 0 | 0.0173 |

Model timings include generation; nonneural timings include sampling/selection and numerical evaluation. They are not equal-compute conditions. Exhaustive search uses a different candidate budget. See the protocol for EOS and prefill accounting.

## Paired descriptive comparisons

Differences are conditioned minus comparator: negative favors conditioning. Intervals resample ten functions after averaging repeats; they are descriptive and do not establish population effects.

| Comparator | Regime | Mean NMSE difference | Descriptive 95% interval |
| --- | --- | ---: | --- |
| no_observations | interpolation | 0.214339 | [0.0359628, 0.442169] |
| no_observations | extrapolation | 31.1551 | [-2.01897, 94.9845] |
| uniform | interpolation | -0.452543 | [-2.23645, 1.31057] |
| uniform | extrapolation | -52.0463 | [-185.114, 48.9756] |
| exhaustive | interpolation | 2.29998 | [0.353043, 5.46445] |
| exhaustive | extrapolation | 46.2784 | [-29.7721, 166.82] |

## Reproducibility and scope

Configuration digest: `d2ab07a55ca8888970e88603e85386547818c5990b921eb563da5e6d6e9d1585`.
Dataset digest: `31b9c4689f75ea9217b12c3c51620ce2a6b0a10455c5f14147880f1efa67e2bf`.

Exact versions, source hashes, model-file hashes, rendered prompts, token IDs, per-candidate scores and task points are retained in the result directory.
This report is generated from those records. Scientific interpretation and smoke-test history are in the experiment README. No private test, weight editing, or claim of new symbolic-search capability is involved.
