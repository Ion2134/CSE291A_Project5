# Sweep results: agent-v9-oldreads

Pool: 1932 rows in 101 request groups, labels {'adversary': 114, 'agent': 95, 'none': 1723}.
Each cell is the **median** over all repeats, with (mean ± std).

| Model | Repeats | cause_macro_f1 | cause_accuracy | error_auroc | error_detection_f1 | false_alarm_rate | missed_error_rate | error_type_accuracy | correctness_brier | correctness_ece |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 2 | 0.326 (0.326 ± 0.000) | 0.958 (0.958 ± 0.000) | 0.500 (0.500 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.042 (0.042 ± 0.000) | 0.042 (0.042 ± 0.000) |
| rules | 2 | 0.441 (0.441 ± 0.000) | 0.875 (0.875 ± 0.000) | 0.914 (0.914 ± 0.000) | 0.389 (0.389 ± 0.000) | 0.128 (0.128 ± 0.000) | 0.043 (0.043 ± 0.000) | 0.957 (0.957 ± 0.000) | 0.122 (0.122 ± 0.000) | 0.097 (0.097 ± 0.000) |
| logreg_checks | 2 | 0.666 (0.666 ± 0.000) | 0.998 (0.998 ± 0.000) | 0.985 (0.985 ± 0.002) | 0.739 (0.739 ± 0.338) | 0.041 (0.041 ± 0.057) | 0.043 (0.043 ± 0.000) | 0.957 (0.957 ± 0.000) | 0.097 (0.097 ± 0.003) | 0.299 (0.299 ± 0.003) |
| gbm_checks | 2 | 0.666 (0.666 ± 0.000) | 0.998 (0.998 ± 0.000) | 0.987 (0.987 ± 0.001) | 0.978 (0.978 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.043 (0.043 ± 0.000) | 0.957 (0.957 ± 0.000) | 0.101 (0.101 ± 0.013) | 0.303 (0.303 ± 0.021) |

## Model vs model on the same repeats (median of first − second)

| First | Second | Metric | Repeats | Median diff | First better |
| --- | --- | --- | ---: | ---: | ---: |
| majority | rules | cause_macro_f1 | 2 | -0.115 | 0% |
| majority | rules | error_auroc | 2 | -0.414 | 0% |
| majority | rules | false_alarm_rate | 2 | -0.128 | 100% |
| majority | logreg_checks | cause_macro_f1 | 2 | -0.340 | 0% |
| majority | logreg_checks | error_auroc | 2 | -0.485 | 0% |
| majority | logreg_checks | false_alarm_rate | 2 | -0.041 | 50% |
| majority | gbm_checks | cause_macro_f1 | 2 | -0.340 | 0% |
| majority | gbm_checks | error_auroc | 2 | -0.487 | 0% |
| majority | gbm_checks | false_alarm_rate | 2 | +0.000 | 0% |
| rules | logreg_checks | cause_macro_f1 | 2 | -0.225 | 0% |
| rules | logreg_checks | error_auroc | 2 | -0.071 | 0% |
| rules | logreg_checks | false_alarm_rate | 2 | +0.088 | 0% |
| rules | gbm_checks | cause_macro_f1 | 2 | -0.225 | 0% |
| rules | gbm_checks | error_auroc | 2 | -0.073 | 0% |
| rules | gbm_checks | false_alarm_rate | 2 | +0.128 | 0% |
| logreg_checks | gbm_checks | cause_macro_f1 | 2 | +0.000 | 0% |
| logreg_checks | gbm_checks | error_auroc | 2 | -0.002 | 0% |
| logreg_checks | gbm_checks | false_alarm_rate | 2 | +0.041 | 0% |

## Caveats

- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, while they use inner cross-validation on all training groups.
- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 (uncalibrated); compare Brier score and ECE with that in mind.
- Each repeat combines the k separately trained fold models.
- Development-scale data: treat these as comparisons, not benchmark claims.
