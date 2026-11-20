# Sweep results: agent-v7-logreg-ngram

Pool: 2174 rows in 104 request groups, labels {'adversary': 178, 'agent': 254, 'none': 1742}.
Each cell is the **median** over all repeats, with (mean ± std).

| Model | Repeats | cause_macro_f1 | cause_accuracy | error_auroc | error_detection_f1 | false_alarm_rate | missed_error_rate | error_type_accuracy | correctness_brier | correctness_ece |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 2 | 0.297 (0.297 ± 0.000) | 0.801 (0.801 ± 0.000) | 0.500 (0.500 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.199 (0.199 ± 0.000) | 0.199 (0.199 ± 0.000) |
| logreg_checks | 2 | 0.689 (0.689 ± 0.006) | 0.750 (0.750 ± 0.002) | 0.878 (0.878 ± 0.006) | 0.606 (0.606 ± 0.000) | 0.269 (0.269 ± 0.000) | 0.095 (0.095 ± 0.000) | 0.825 (0.825 ± 0.008) | 0.183 (0.183 ± 0.004) | 0.180 (0.180 ± 0.029) |
| logreg_memory | 2 | 0.745 (0.745 ± 0.003) | 0.768 (0.768 ± 0.001) | 0.924 (0.924 ± 0.005) | 0.636 (0.636 ± 0.000) | 0.269 (0.269 ± 0.000) | 0.028 (0.028 ± 0.000) | 0.916 (0.916 ± 0.005) | 0.144 (0.144 ± 0.008) | 0.119 (0.119 ± 0.011) |

## Model vs model on the same repeats (median of first − second)

| First | Second | Metric | Repeats | Median diff | First better |
| --- | --- | --- | ---: | ---: | ---: |
| majority | logreg_checks | cause_macro_f1 | 2 | -0.392 | 0% |
| majority | logreg_checks | error_auroc | 2 | -0.378 | 0% |
| majority | logreg_checks | false_alarm_rate | 2 | -0.269 | 100% |
| majority | logreg_memory | cause_macro_f1 | 2 | -0.448 | 0% |
| majority | logreg_memory | error_auroc | 2 | -0.424 | 0% |
| majority | logreg_memory | false_alarm_rate | 2 | -0.269 | 100% |
| logreg_checks | logreg_memory | cause_macro_f1 | 2 | -0.056 | 0% |
| logreg_checks | logreg_memory | error_auroc | 2 | -0.046 | 0% |
| logreg_checks | logreg_memory | false_alarm_rate | 2 | +0.000 | 0% |

## Caveats

- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, while they use inner cross-validation on all training groups.
- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 (uncalibrated); compare Brier score and ECE with that in mind.
- Each repeat combines the k separately trained fold models.
- Development-scale data: treat these as comparisons, not benchmark claims.
