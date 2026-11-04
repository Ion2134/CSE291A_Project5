# Sweep results: agent-v4

Pool: 169 rows in 10 request groups, labels {'adversary': 32, 'agent': 16, 'none': 121}.
Each cell is the **median** over all repeats, with (mean ± std).

| Model | Repeats | cause_macro_f1 | cause_accuracy | error_auroc | error_detection_f1 | false_alarm_rate | missed_error_rate | error_type_accuracy | correctness_brier | correctness_ece |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 5 | 0.278 (0.278 ± 0.000) | 0.716 (0.716 ± 0.000) | 0.500 (0.500 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.284 (0.284 ± 0.000) | 0.284 (0.284 ± 0.000) |
| mlp | 15 | 0.666 (0.665 ± 0.028) | 0.781 (0.782 ± 0.018) | 0.910 (0.904 ± 0.017) | 0.741 (0.735 ± 0.021) | 0.174 (0.170 ± 0.018) | 0.167 (0.171 ± 0.021) | 0.646 (0.640 ± 0.029) | 0.125 (0.124 ± 0.008) | 0.077 (0.075 ± 0.015) |
| stacking | 15 | 0.603 (0.609 ± 0.017) | 0.716 (0.721 ± 0.014) | 0.868 (0.866 ± 0.015) | 0.662 (0.655 ± 0.013) | 0.355 (0.353 ± 0.022) | 0.083 (0.079 ± 0.009) | 0.646 (0.637 ± 0.022) | 0.189 (0.192 ± 0.007) | 0.086 (0.087 ± 0.006) |
| laya | 1 | 0.690 (0.690 ± 0.000) | 0.870 (0.870 ± 0.000) | 0.976 (0.976 ± 0.000) | 0.831 (0.831 ± 0.000) | 0.033 (0.033 ± 0.000) | 0.229 (0.229 ± 0.000) | 0.562 (0.562 ± 0.000) | 0.053 (0.053 ± 0.000) | 0.041 (0.041 ± 0.000) |

## Model vs model on the same repeats (median of first − second)

| First | Second | Metric | Repeats | Median diff | First better |
| --- | --- | --- | ---: | ---: | ---: |
| majority | mlp | cause_macro_f1 | 15 | -0.387 | 0% |
| majority | mlp | error_auroc | 15 | -0.410 | 0% |
| majority | mlp | false_alarm_rate | 15 | -0.174 | 100% |
| majority | stacking | cause_macro_f1 | 15 | -0.325 | 0% |
| majority | stacking | error_auroc | 15 | -0.368 | 0% |
| majority | stacking | false_alarm_rate | 15 | -0.355 | 100% |
| majority | laya | cause_macro_f1 | 1 | -0.412 | 0% |
| majority | laya | error_auroc | 1 | -0.476 | 0% |
| majority | laya | false_alarm_rate | 1 | -0.033 | 100% |
| mlp | stacking | cause_macro_f1 | 15 | +0.066 | 93% |
| mlp | stacking | error_auroc | 15 | +0.040 | 100% |
| mlp | stacking | false_alarm_rate | 15 | -0.190 | 100% |
| mlp | laya | cause_macro_f1 | 1 | +0.022 | 100% |
| mlp | laya | error_auroc | 1 | -0.054 | 0% |
| mlp | laya | false_alarm_rate | 1 | +0.124 | 0% |
| stacking | laya | cause_macro_f1 | 1 | -0.089 | 0% |
| stacking | laya | error_auroc | 1 | -0.108 | 0% |
| stacking | laya | false_alarm_rate | 1 | +0.322 | 0% |

## Caveats

- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, while they use inner cross-validation on all training groups.
- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 (uncalibrated); compare Brier score and ECE with that in mind.
- Each repeat combines the k separately trained fold models.
- Development-scale data: treat these as comparisons, not benchmark claims.
