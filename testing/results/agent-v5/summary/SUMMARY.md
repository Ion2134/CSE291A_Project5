# Sweep results: agent-v5

Pool: 580 rows in 35 request groups, labels {'adversary': 33, 'agent': 160, 'none': 387}.
Each cell is the **median** over all repeats, with (mean ± std).

| Model | Repeats | cause_macro_f1 | cause_accuracy | error_auroc | error_detection_f1 | false_alarm_rate | missed_error_rate | error_type_accuracy | correctness_brier | correctness_ece |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 5 | 0.267 (0.267 ± 0.000) | 0.667 (0.667 ± 0.000) | 0.500 (0.500 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.333 (0.333 ± 0.000) | 0.333 (0.333 ± 0.000) |
| mlp | 15 | 0.684 (0.684 ± 0.015) | 0.740 (0.739 ± 0.012) | 0.862 (0.859 ± 0.010) | 0.691 (0.691 ± 0.008) | 0.292 (0.297 ± 0.032) | 0.166 (0.158 ± 0.039) | 0.782 (0.784 ± 0.035) | 0.162 (0.161 ± 0.006) | 0.050 (0.054 ± 0.012) |
| stacking | 15 | 0.642 (0.636 ± 0.016) | 0.684 (0.683 ± 0.009) | 0.844 (0.843 ± 0.007) | 0.675 (0.675 ± 0.010) | 0.419 (0.422 ± 0.022) | 0.062 (0.060 ± 0.003) | 0.850 (0.845 ± 0.014) | 0.195 (0.197 ± 0.008) | 0.044 (0.048 ± 0.018) |
| laya | 3 | 0.609 (0.619 ± 0.032) | 0.767 (0.764 ± 0.005) | 0.885 (0.879 ± 0.010) | 0.681 (0.684 ± 0.016) | 0.132 (0.132 ± 0.005) | 0.352 (0.342 ± 0.027) | 0.518 (0.532 ± 0.038) | 0.131 (0.133 ± 0.005) | 0.024 (0.024 ± 0.010) |

## Model vs model on the same repeats (median of first − second)

| First | Second | Metric | Repeats | Median diff | First better |
| --- | --- | --- | ---: | ---: | ---: |
| majority | mlp | cause_macro_f1 | 15 | -0.417 | 0% |
| majority | mlp | error_auroc | 15 | -0.362 | 0% |
| majority | mlp | false_alarm_rate | 15 | -0.292 | 100% |
| majority | stacking | cause_macro_f1 | 15 | -0.375 | 0% |
| majority | stacking | error_auroc | 15 | -0.344 | 0% |
| majority | stacking | false_alarm_rate | 15 | -0.419 | 100% |
| majority | laya | cause_macro_f1 | 3 | -0.342 | 0% |
| majority | laya | error_auroc | 3 | -0.385 | 0% |
| majority | laya | false_alarm_rate | 3 | -0.132 | 100% |
| mlp | stacking | cause_macro_f1 | 15 | +0.051 | 100% |
| mlp | stacking | error_auroc | 15 | +0.014 | 100% |
| mlp | stacking | false_alarm_rate | 15 | -0.116 | 100% |
| mlp | laya | cause_macro_f1 | 3 | +0.082 | 100% |
| mlp | laya | error_auroc | 3 | -0.022 | 33% |
| mlp | laya | false_alarm_rate | 3 | +0.155 | 0% |
| stacking | laya | cause_macro_f1 | 3 | +0.021 | 67% |
| stacking | laya | error_auroc | 3 | -0.033 | 0% |
| stacking | laya | false_alarm_rate | 3 | +0.287 | 0% |

## Caveats

- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, while they use inner cross-validation on all training groups.
- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 (uncalibrated); compare Brier score and ECE with that in mind.
- Each repeat combines the k separately trained fold models.
- Development-scale data: treat these as comparisons, not benchmark claims.
