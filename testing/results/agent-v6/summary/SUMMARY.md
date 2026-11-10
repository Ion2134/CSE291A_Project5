# Sweep results: agent-v6

Pool: 580 rows in 35 request groups, labels {'adversary': 33, 'agent': 160, 'none': 387}.
Each cell is the **median** over all repeats, with (mean ± std).

| Model | Repeats | cause_macro_f1 | cause_accuracy | error_auroc | error_detection_f1 | false_alarm_rate | missed_error_rate | error_type_accuracy | correctness_brier | correctness_ece |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 5 | 0.267 (0.267 ± 0.000) | 0.667 (0.667 ± 0.000) | 0.500 (0.500 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.333 (0.333 ± 0.000) | 0.333 (0.333 ± 0.000) |
| mlp | 15 | 0.981 (0.979 ± 0.005) | 0.995 (0.994 ± 0.001) | 1.000 (1.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.984 (0.982 ± 0.004) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) |
| stacking | 15 | 0.981 (0.981 ± 0.000) | 0.995 (0.995 ± 0.000) | 1.000 (1.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.984 (0.984 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) |
| rules | 5 | 0.981 (0.981 ± 0.000) | 0.995 (0.995 ± 0.000) | 1.000 (1.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.984 (0.984 ± 0.000) | 0.001 (0.001 ± 0.000) | 0.025 (0.025 ± 0.000) |

## Model vs model on the same repeats (median of first − second)

| First | Second | Metric | Repeats | Median diff | First better |
| --- | --- | --- | ---: | ---: | ---: |
| majority | mlp | cause_macro_f1 | 15 | -0.714 | 0% |
| majority | mlp | error_auroc | 15 | -0.500 | 0% |
| majority | mlp | false_alarm_rate | 15 | +0.000 | 0% |
| majority | stacking | cause_macro_f1 | 15 | -0.714 | 0% |
| majority | stacking | error_auroc | 15 | -0.500 | 0% |
| majority | stacking | false_alarm_rate | 15 | +0.000 | 0% |
| majority | rules | cause_macro_f1 | 5 | -0.714 | 0% |
| majority | rules | error_auroc | 5 | -0.500 | 0% |
| majority | rules | false_alarm_rate | 5 | +0.000 | 0% |
| mlp | stacking | cause_macro_f1 | 15 | +0.000 | 0% |
| mlp | stacking | error_auroc | 15 | +0.000 | 0% |
| mlp | stacking | false_alarm_rate | 15 | +0.000 | 0% |
| mlp | rules | cause_macro_f1 | 5 | +0.000 | 0% |
| mlp | rules | error_auroc | 5 | +0.000 | 0% |
| mlp | rules | false_alarm_rate | 5 | +0.000 | 0% |
| stacking | rules | cause_macro_f1 | 5 | +0.000 | 0% |
| stacking | rules | error_auroc | 5 | +0.000 | 0% |
| stacking | rules | false_alarm_rate | 5 | +0.000 | 0% |

## Caveats

- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, while they use inner cross-validation on all training groups.
- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 (uncalibrated); compare Brier score and ECE with that in mind.
- Each repeat combines the k separately trained fold models.
- Development-scale data: treat these as comparisons, not benchmark claims.
