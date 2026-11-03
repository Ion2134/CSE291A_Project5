# Sweep results: agent-v4

Pool: 169 rows in 10 request groups, labels {'adversary': 32, 'agent': 16, 'none': 121}.
Each cell is the **median** over all repeats, with (mean ± std).

| Model | Repeats | cause_macro_f1 | cause_accuracy | error_auroc | error_detection_f1 | false_alarm_rate | missed_error_rate | error_type_accuracy | correctness_brier | correctness_ece |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 5 | 0.278 (0.278 ± 0.000) | 0.716 (0.716 ± 0.000) | 0.500 (0.500 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.284 (0.284 ± 0.000) | 0.284 (0.284 ± 0.000) |
| mlp | 15 | 0.666 (0.665 ± 0.028) | 0.781 (0.782 ± 0.018) | 0.910 (0.904 ± 0.017) | 0.741 (0.735 ± 0.021) | 0.174 (0.170 ± 0.018) | 0.167 (0.171 ± 0.021) | 0.646 (0.640 ± 0.029) | 0.125 (0.124 ± 0.008) | 0.077 (0.075 ± 0.015) |

## Model vs model on the same repeats (median of first − second)

| First | Second | Metric | Repeats | Median diff | First better |
| --- | --- | --- | ---: | ---: | ---: |
| majority | mlp | cause_macro_f1 | 15 | -0.387 | 0% |
| majority | mlp | error_auroc | 15 | -0.410 | 0% |
| majority | mlp | false_alarm_rate | 15 | -0.174 | 100% |

## Caveats

- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, while they use inner cross-validation on all training groups.
- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 (uncalibrated); compare Brier score and ECE with that in mind.
- Each repeat combines the k separately trained fold models.
- Development-scale data: treat these as comparisons, not benchmark claims.
