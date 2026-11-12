# Baseline diagnostic suite: executed results

These are exact finite-catalog distributions on public development/evaluation tasks. No private final test or structural synthesis claim is involved. Expected best-of-eight selects using original observation scores; reserved errors never choose expressions.

## Frozen development selections

| Model | Selected prompt | Minimal regret | Catalog regret | Few-shot regret |
| --- | --- | ---: | ---: | ---: |
| 135m | catalog | 1.75992 | 1.57556 | 2.60918 |
| 360m | catalog | 1.53388 | 0.494138 | 1.30309 |

## Evaluation: correct observations, representable functions

| Model | Prompt | Target probability | Expected target hit at K=8 | Expected interpolation NMSE | Expected extrapolation NMSE | Observation regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 135m | minimal | 0.0895097 | 20.62% | 1.13713 | 117.635 | 1.0381 |
| 135m | catalog | 0.0900819 | 30.68% | 1.16116 | 22.1674 | 0.958267 |
| 135m | few_shot | 0.064666 | 22.41% | 1.5739 | 227.072 | 1.37104 |
| 360m | minimal | 0.0330505 | 21.73% | 4.89063 | 329.326 | 5.24459 |
| 360m | catalog | 0.0965044 | 44.35% | 4.41523 | 143.294 | 4.12822 |
| 360m | few_shot | 0.0565 | 29.65% | 4.45152 | 117.808 | 4.85915 |

## Observation controls under the development-selected prompt

| Model | Control | Target probability | Expected interpolation NMSE | Expected extrapolation NMSE | Observation regret |
| --- | --- | ---: | ---: | ---: | ---: |
| 135m | correct | 0.0900819 | 1.16116 | 22.1674 | 0.958267 |
| 135m | shuffled | 0.093448 | 1.15295 | 18.5253 | 0.954302 |
| 135m | other_function | 0.108041 | 1.2042 | 15.7512 | 0.970813 |
| 135m | absent | 0.113454 | 1.07384 | 22.389 | 0.874578 |
| 360m | correct | 0.0965044 | 4.41523 | 143.294 | 4.12822 |
| 360m | shuffled | 0.0928697 | 4.41061 | 156.261 | 4.13697 |
| 360m | other_function | 0.0938876 | 4.33742 | 152.461 | 4.0688 |
| 360m | absent | 0.0823128 | 2.53386 | 68.6197 | 2.38681 |

## Raw MSE under the development-selected prompt

MSE complements NMSE; target variance differs across functions and domains.

| Model | Control | Expected interpolation MSE | Expected extrapolation MSE |
| --- | --- | ---: | ---: |
| 135m | correct | 0.163188 | 3.18543 |
| 135m | shuffled | 0.158846 | 3.10653 |
| 135m | other_function | 0.156601 | 2.88782 |
| 135m | absent | 0.169721 | 3.61292 |
| 360m | correct | 0.12491 | 1.96159 |
| 360m | shuffled | 0.126175 | 1.99319 |
| 360m | other_function | 0.133564 | 1.93751 |
| 360m | absent | 0.109552 | 1.61652 |

## Per-function target probability

| Target | 135M minimal | 135M catalog | 135M few-shot | 360M minimal | 360M catalog | 360M few-shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sin(x)` | 0.000476949 | 0.0745089 | 0.0173927 | 0.00178196 | 0.127935 | 0.188527 |
| `cos(x)` | 0.000150814 | 0.0104136 | 0.00321898 | 0.00312912 | 0.0250647 | 0.0424547 |
| `x*x` | 0.00136428 | 2.9716e-07 | 3.67511e-06 | 0.042146 | 0.264403 | 0.0259051 |
| `exp(x)` | 0.00307514 | 0.00238944 | 0.00271768 | 0.0394047 | 0.0483706 | 0.0253625 |
| `1/(x*x+1)` | 0.442481 | 0.363097 | 0.299997 | 0.0787908 | 0.0167494 | 0.000250745 |

## Analytical nonneural references

Uniform uses K=8 draws with replacement. Exhaustive selection evaluates all 14 entries. Both use the same correct observations and complexity penalty. These are derived without neural inference.

| Group | Reference | Expected interpolation NMSE | Expected extrapolation NMSE | Expected observation regret |
| --- | --- | ---: | ---: | ---: |
| representable | uniform | 1.6644 | 61.0233 | 1.56702 |
| representable | exhaustive | 0 | 0 | 0 |
| outside_catalog | uniform | 12.3421 | 184.492 | 1.92256 |
| outside_catalog | exhaustive | 9.72745 | 58.9167 | 0 |

## Outside-catalog tasks, correct observations

| Model | Prompt | Expected interpolation NMSE | Expected extrapolation NMSE | Observation regret |
| --- | --- | ---: | ---: | ---: |
| 135m | catalog | 12.8481 | 40.1132 | 1.29209 |
| 360m | catalog | 12.4272 | 560.626 | 2.56597 |

## Targeted intervention

Status: `completed`.

Development-selected candidate: `rank_one_0.001`. Development screen passed: **False**.
Development baseline regret 1.57556; candidate regret 2.8455.

A failed screen remains a rejected diagnostic candidate; evaluation does not promote it.

| Condition | Expected interpolation NMSE | Expected extrapolation NMSE | Target probability | Observation regret |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.16116 | 22.1674 | 0.0900819 | 0.958267 |
| targeted | 13.4414 | 45.785 | 0.119694 | 10.3605 |
| no_edit | 1.16116 | 22.1674 | 0.0900819 | 0.958267 |
| root_logit_bias | 2.53352 | 38.0801 | 0.0611516 | 2.20565 |
| random_71 | 1.16242 | 22.1464 | 0.0899345 | 0.959448 |
| random_72 | 1.16175 | 22.1544 | 0.0899736 | 0.959102 |

### All development edit proposals

| Candidate | Relative tensor norm | Development regret |
| --- | ---: | ---: |
| rank_one_0.001 | 0.100% | 2.8455 |
| rank_one_0.01 | 1.000% | 9.49612 |

Original tensor restored and hash-checked: **True**.

## Distribution sensitivity under the selected prompt

Representable functions only. Positive target-probability differences favor correct data; negative error differences favor correct data. Total variation measures change, not improvement.

| Model | Correct versus | Mean total variation | Mean target-probability difference | Mean interpolation-NMSE difference |
| --- | --- | ---: | ---: | ---: |
| 135m | shuffled | 0.0293946 | -0.0033661 | 0.00821168 |
| 135m | other_function | 0.1038 | -0.0179589 | -0.0430342 |
| 135m | absent | 0.116513 | -0.0233717 | 0.0873192 |
| 360m | shuffled | 0.0368388 | 0.00363474 | 0.00461902 |
| 360m | other_function | 0.11064 | 0.00261683 | 0.0778095 |
| 360m | absent | 0.390178 | 0.0141916 | 1.88138 |

## Accounting and limits

Completed matrix rows: 270. Probability mass is checked at every catalog distribution. Sum of recorded noncached matrix inference times: 1331.9 s.
Repeated identical prompts reuse exact distributions and are marked in the traces. Intervention work is separate; see its per-row timings. Wall timings are shared-CPU observations, not controlled latency benchmarks.
Each group contains only five functions and one numerical realization per function. Exact probabilities remove Monte Carlo sampling uncertainty, not task/population uncertainty. No confirmatory significance claim or scaling law is established.
All three prompt variants and all observation controls are reported; prompt choice and edit magnitude use development data only. Evaluation labels and reserved points never enter the generator. Historical baseline functions are reused with new points, so this is not an independent final test.
Source/model hashes, effective prompts, prefix log probabilities, selection probabilities and per-function errors are retained in the raw artifacts. The README discusses scientific interpretation and implementation details.
