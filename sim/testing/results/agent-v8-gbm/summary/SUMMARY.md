# Sweep results: agent-v8-gbm

Pool: 2174 rows in 104 request groups, labels {'adversary': 178, 'agent': 254, 'none': 1742}.
Each cell is the **median** over all repeats, with (mean ± std).

| Model | Repeats | cause_macro_f1 | cause_accuracy | error_auroc | error_detection_f1 | false_alarm_rate | missed_error_rate | error_type_accuracy | correctness_brier | correctness_ece |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 2 | 0.297 (0.297 ± 0.000) | 0.801 (0.801 ± 0.000) | 0.500 (0.500 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.199 (0.199 ± 0.000) | 0.199 (0.199 ± 0.000) |
| logreg_checks | 2 | 0.831 (0.831 ± 0.006) | 0.924 (0.924 ± 0.001) | 0.915 (0.915 ± 0.024) | 0.803 (0.803 ± 0.037) | 0.058 (0.058 ± 0.024) | 0.175 (0.175 ± 0.002) | 0.782 (0.782 ± 0.007) | 0.106 (0.106 ± 0.005) | 0.177 (0.177 ± 0.004) |
| logreg_memory | 2 | 0.875 (0.875 ± 0.004) | 0.941 (0.941 ± 0.001) | 0.974 (0.974 ± 0.006) | 0.819 (0.819 ± 0.006) | 0.080 (0.080 ± 0.008) | 0.082 (0.082 ± 0.011) | 0.866 (0.866 ± 0.007) | 0.058 (0.058 ± 0.003) | 0.059 (0.059 ± 0.013) |
| gbm_checks | 2 | 0.835 (0.835 ± 0.000) | 0.925 (0.925 ± 0.000) | 0.916 (0.916 ± 0.021) | 0.772 (0.772 ± 0.005) | 0.080 (0.080 ± 0.008) | 0.168 (0.168 ± 0.011) | 0.785 (0.785 ± 0.000) | 0.102 (0.102 ± 0.004) | 0.118 (0.118 ± 0.008) |
| gbm_memory | 2 | 0.874 (0.874 ± 0.006) | 0.940 (0.940 ± 0.002) | 0.973 (0.973 ± 0.007) | 0.818 (0.818 ± 0.006) | 0.080 (0.080 ± 0.008) | 0.084 (0.084 ± 0.011) | 0.862 (0.862 ± 0.008) | 0.056 (0.056 ± 0.003) | 0.044 (0.044 ± 0.018) |

## Model vs model on the same repeats (median of first − second)

| First | Second | Metric | Repeats | Median diff | First better |
| --- | --- | --- | ---: | ---: | ---: |
| majority | logreg_checks | cause_macro_f1 | 2 | -0.535 | 0% |
| majority | logreg_checks | error_auroc | 2 | -0.415 | 0% |
| majority | logreg_checks | false_alarm_rate | 2 | -0.058 | 100% |
| majority | logreg_memory | cause_macro_f1 | 2 | -0.579 | 0% |
| majority | logreg_memory | error_auroc | 2 | -0.474 | 0% |
| majority | logreg_memory | false_alarm_rate | 2 | -0.080 | 100% |
| majority | gbm_checks | cause_macro_f1 | 2 | -0.538 | 0% |
| majority | gbm_checks | error_auroc | 2 | -0.416 | 0% |
| majority | gbm_checks | false_alarm_rate | 2 | -0.080 | 100% |
| majority | gbm_memory | cause_macro_f1 | 2 | -0.577 | 0% |
| majority | gbm_memory | error_auroc | 2 | -0.473 | 0% |
| majority | gbm_memory | false_alarm_rate | 2 | -0.080 | 100% |
| logreg_checks | logreg_memory | cause_macro_f1 | 2 | -0.044 | 0% |
| logreg_checks | logreg_memory | error_auroc | 2 | -0.059 | 0% |
| logreg_checks | logreg_memory | false_alarm_rate | 2 | -0.023 | 50% |
| logreg_checks | gbm_checks | cause_macro_f1 | 2 | -0.003 | 50% |
| logreg_checks | gbm_checks | error_auroc | 2 | -0.001 | 50% |
| logreg_checks | gbm_checks | false_alarm_rate | 2 | -0.023 | 50% |
| logreg_checks | gbm_memory | cause_macro_f1 | 2 | -0.042 | 0% |
| logreg_checks | gbm_memory | error_auroc | 2 | -0.058 | 0% |
| logreg_checks | gbm_memory | false_alarm_rate | 2 | -0.023 | 50% |
| logreg_memory | gbm_checks | cause_macro_f1 | 2 | +0.040 | 100% |
| logreg_memory | gbm_checks | error_auroc | 2 | +0.058 | 100% |
| logreg_memory | gbm_checks | false_alarm_rate | 2 | +0.000 | 0% |
| logreg_memory | gbm_memory | cause_macro_f1 | 2 | +0.002 | 50% |
| logreg_memory | gbm_memory | error_auroc | 2 | +0.001 | 100% |
| logreg_memory | gbm_memory | false_alarm_rate | 2 | +0.000 | 0% |
| gbm_checks | gbm_memory | cause_macro_f1 | 2 | -0.039 | 0% |
| gbm_checks | gbm_memory | error_auroc | 2 | -0.057 | 0% |
| gbm_checks | gbm_memory | false_alarm_rate | 2 | +0.000 | 0% |

## Caveats

- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, while they use inner cross-validation on all training groups.
- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 (uncalibrated); compare Brier score and ECE with that in mind.
- Each repeat combines the k separately trained fold models.
- Development-scale data: treat these as comparisons, not benchmark claims.
