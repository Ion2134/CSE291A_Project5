# Sweep results: agent-v6

Pool: 580 rows in 35 request groups, labels {'adversary': 33, 'agent': 160, 'none': 387}.
Each cell is the **median** over all repeats, with (mean ± std).

| Model | Repeats | cause_macro_f1 | cause_accuracy | error_auroc | error_detection_f1 | false_alarm_rate | missed_error_rate | error_type_accuracy | correctness_brier | correctness_ece |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 2 | 0.267 (0.267 ± 0.000) | 0.667 (0.667 ± 0.000) | 0.500 (0.500 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.333 (0.333 ± 0.000) | 0.333 (0.333 ± 0.000) |
| mlp | 6 | 0.904 (0.905 ± 0.012) | 0.976 (0.976 ± 0.003) | 0.999 (0.999 ± 0.000) | 0.971 (0.971 ± 0.004) | 0.003 (0.002 ± 0.001) | 0.054 (0.054 ± 0.006) | 0.930 (0.931 ± 0.006) | 0.015 (0.015 ± 0.002) | 0.013 (0.014 ± 0.002) |
| mlp_embed | 6 | 0.891 (0.894 ± 0.017) | 0.966 (0.967 ± 0.004) | 0.997 (0.996 ± 0.002) | 0.956 (0.958 ± 0.006) | 0.025 (0.026 ± 0.004) | 0.034 (0.032 ± 0.016) | 0.951 (0.953 ± 0.016) | 0.023 (0.022 ± 0.004) | 0.019 (0.019 ± 0.008) |
| stacking | 6 | 0.939 (0.939 ± 0.003) | 0.984 (0.984 ± 0.001) | 0.998 (0.998 ± 0.000) | 0.983 (0.983 ± 0.001) | 0.001 (0.001 ± 0.001) | 0.031 (0.031 ± 0.000) | 0.953 (0.953 ± 0.000) | 0.010 (0.010 ± 0.001) | 0.006 (0.006 ± 0.001) |
| rules | 2 | 0.942 (0.942 ± 0.000) | 0.984 (0.984 ± 0.000) | 0.984 (0.984 ± 0.000) | 0.984 (0.984 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.031 (0.031 ± 0.000) | 0.953 (0.953 ± 0.000) | 0.010 (0.010 ± 0.000) | 0.015 (0.015 ± 0.000) |
| rules_plus | 2 | 1.000 (1.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.001 (0.001 ± 0.000) | 0.025 (0.025 ± 0.000) |
| knn | 2 | 0.833 (0.833 ± 0.025) | 0.927 (0.927 ± 0.011) | 0.998 (0.998 ± 0.001) | 0.926 (0.926 ± 0.007) | 0.076 (0.076 ± 0.013) | 0.005 (0.005 ± 0.007) | 0.933 (0.933 ± 0.059) | 0.035 (0.035 ± 0.003) | 0.102 (0.102 ± 0.007) |
| knn_embed | 2 | 0.823 (0.823 ± 0.012) | 0.922 (0.922 ± 0.002) | 0.989 (0.989 ± 0.009) | 0.900 (0.900 ± 0.004) | 0.098 (0.098 ± 0.004) | 0.021 (0.021 ± 0.015) | 0.964 (0.964 ± 0.015) | 0.046 (0.046 ± 0.004) | 0.107 (0.107 ± 0.004) |
| logreg_checks | 2 | 0.942 (0.942 ± 0.000) | 0.984 (0.984 ± 0.000) | 0.990 (0.990 ± 0.000) | 0.984 (0.984 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.031 (0.031 ± 0.000) | 0.953 (0.953 ± 0.000) | 0.034 (0.034 ± 0.000) | 0.128 (0.128 ± 0.001) |
| logreg_memory | 2 | 0.981 (0.981 ± 0.000) | 0.995 (0.995 ± 0.000) | 1.000 (1.000 ± 0.000) | 1.000 (1.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.000 (0.000 ± 0.000) | 0.984 (0.984 ± 0.000) | 0.005 (0.005 ± 0.000) | 0.066 (0.066 ± 0.000) |

## Model vs model on the same repeats (median of first − second)

| First | Second | Metric | Repeats | Median diff | First better |
| --- | --- | --- | ---: | ---: | ---: |
| majority | mlp | cause_macro_f1 | 6 | -0.637 | 0% |
| majority | mlp | error_auroc | 6 | -0.499 | 0% |
| majority | mlp | false_alarm_rate | 6 | -0.003 | 67% |
| majority | mlp_embed | cause_macro_f1 | 6 | -0.624 | 0% |
| majority | mlp_embed | error_auroc | 6 | -0.497 | 0% |
| majority | mlp_embed | false_alarm_rate | 6 | -0.025 | 100% |
| majority | stacking | cause_macro_f1 | 6 | -0.672 | 0% |
| majority | stacking | error_auroc | 6 | -0.498 | 0% |
| majority | stacking | false_alarm_rate | 6 | -0.001 | 50% |
| majority | rules | cause_macro_f1 | 2 | -0.675 | 0% |
| majority | rules | error_auroc | 2 | -0.484 | 0% |
| majority | rules | false_alarm_rate | 2 | +0.000 | 0% |
| majority | rules_plus | cause_macro_f1 | 2 | -0.733 | 0% |
| majority | rules_plus | error_auroc | 2 | -0.500 | 0% |
| majority | rules_plus | false_alarm_rate | 2 | +0.000 | 0% |
| majority | knn | cause_macro_f1 | 2 | -0.566 | 0% |
| majority | knn | error_auroc | 2 | -0.498 | 0% |
| majority | knn | false_alarm_rate | 2 | -0.076 | 100% |
| majority | knn_embed | cause_macro_f1 | 2 | -0.556 | 0% |
| majority | knn_embed | error_auroc | 2 | -0.489 | 0% |
| majority | knn_embed | false_alarm_rate | 2 | -0.098 | 100% |
| majority | logreg_checks | cause_macro_f1 | 2 | -0.675 | 0% |
| majority | logreg_checks | error_auroc | 2 | -0.490 | 0% |
| majority | logreg_checks | false_alarm_rate | 2 | +0.000 | 0% |
| majority | logreg_memory | cause_macro_f1 | 2 | -0.714 | 0% |
| majority | logreg_memory | error_auroc | 2 | -0.500 | 0% |
| majority | logreg_memory | false_alarm_rate | 2 | +0.000 | 0% |
| mlp | mlp_embed | cause_macro_f1 | 6 | +0.014 | 50% |
| mlp | mlp_embed | error_auroc | 6 | +0.002 | 83% |
| mlp | mlp_embed | false_alarm_rate | 6 | -0.023 | 100% |
| mlp | stacking | cause_macro_f1 | 6 | -0.035 | 0% |
| mlp | stacking | error_auroc | 6 | +0.000 | 67% |
| mlp | stacking | false_alarm_rate | 6 | +0.001 | 33% |
| mlp | rules | cause_macro_f1 | 2 | -0.042 | 0% |
| mlp | rules | error_auroc | 2 | +0.014 | 100% |
| mlp | rules | false_alarm_rate | 2 | +0.001 | 0% |
| mlp | rules_plus | cause_macro_f1 | 2 | -0.100 | 0% |
| mlp | rules_plus | error_auroc | 2 | -0.001 | 0% |
| mlp | rules_plus | false_alarm_rate | 2 | +0.001 | 0% |
| mlp | knn | cause_macro_f1 | 2 | +0.067 | 100% |
| mlp | knn | error_auroc | 2 | +0.000 | 50% |
| mlp | knn | false_alarm_rate | 2 | -0.075 | 100% |
| mlp | knn_embed | cause_macro_f1 | 2 | +0.077 | 100% |
| mlp | knn_embed | error_auroc | 2 | +0.010 | 100% |
| mlp | knn_embed | false_alarm_rate | 2 | -0.097 | 100% |
| mlp | logreg_checks | cause_macro_f1 | 2 | -0.042 | 0% |
| mlp | logreg_checks | error_auroc | 2 | +0.009 | 100% |
| mlp | logreg_checks | false_alarm_rate | 2 | +0.001 | 0% |
| mlp | logreg_memory | cause_macro_f1 | 2 | -0.081 | 0% |
| mlp | logreg_memory | error_auroc | 2 | -0.001 | 0% |
| mlp | logreg_memory | false_alarm_rate | 2 | +0.001 | 0% |
| mlp_embed | stacking | cause_macro_f1 | 6 | -0.048 | 0% |
| mlp_embed | stacking | error_auroc | 6 | -0.002 | 17% |
| mlp_embed | stacking | false_alarm_rate | 6 | +0.023 | 0% |
| mlp_embed | rules | cause_macro_f1 | 2 | -0.045 | 0% |
| mlp_embed | rules | error_auroc | 2 | +0.012 | 100% |
| mlp_embed | rules | false_alarm_rate | 2 | +0.025 | 0% |
| mlp_embed | rules_plus | cause_macro_f1 | 2 | -0.103 | 0% |
| mlp_embed | rules_plus | error_auroc | 2 | -0.004 | 0% |
| mlp_embed | rules_plus | false_alarm_rate | 2 | +0.025 | 0% |
| mlp_embed | knn | cause_macro_f1 | 2 | +0.064 | 100% |
| mlp_embed | knn | error_auroc | 2 | -0.002 | 50% |
| mlp_embed | knn | false_alarm_rate | 2 | -0.052 | 100% |
| mlp_embed | knn_embed | cause_macro_f1 | 2 | +0.073 | 100% |
| mlp_embed | knn_embed | error_auroc | 2 | +0.008 | 100% |
| mlp_embed | knn_embed | false_alarm_rate | 2 | -0.074 | 100% |
| mlp_embed | logreg_checks | cause_macro_f1 | 2 | -0.045 | 0% |
| mlp_embed | logreg_checks | error_auroc | 2 | +0.007 | 100% |
| mlp_embed | logreg_checks | false_alarm_rate | 2 | +0.025 | 0% |
| mlp_embed | logreg_memory | cause_macro_f1 | 2 | -0.084 | 0% |
| mlp_embed | logreg_memory | error_auroc | 2 | -0.004 | 0% |
| mlp_embed | logreg_memory | false_alarm_rate | 2 | +0.025 | 0% |
| stacking | rules | cause_macro_f1 | 2 | -0.003 | 0% |
| stacking | rules | error_auroc | 2 | +0.014 | 100% |
| stacking | rules | false_alarm_rate | 2 | +0.001 | 0% |
| stacking | rules_plus | cause_macro_f1 | 2 | -0.061 | 0% |
| stacking | rules_plus | error_auroc | 2 | -0.002 | 0% |
| stacking | rules_plus | false_alarm_rate | 2 | +0.001 | 0% |
| stacking | knn | cause_macro_f1 | 2 | +0.106 | 100% |
| stacking | knn | error_auroc | 2 | +0.000 | 50% |
| stacking | knn | false_alarm_rate | 2 | -0.075 | 100% |
| stacking | knn_embed | cause_macro_f1 | 2 | +0.116 | 100% |
| stacking | knn_embed | error_auroc | 2 | +0.010 | 100% |
| stacking | knn_embed | false_alarm_rate | 2 | -0.097 | 100% |
| stacking | logreg_checks | cause_macro_f1 | 2 | -0.003 | 0% |
| stacking | logreg_checks | error_auroc | 2 | +0.009 | 100% |
| stacking | logreg_checks | false_alarm_rate | 2 | +0.001 | 0% |
| stacking | logreg_memory | cause_macro_f1 | 2 | -0.042 | 0% |
| stacking | logreg_memory | error_auroc | 2 | -0.002 | 0% |
| stacking | logreg_memory | false_alarm_rate | 2 | +0.001 | 0% |
| rules | rules_plus | cause_macro_f1 | 2 | -0.058 | 0% |
| rules | rules_plus | error_auroc | 2 | -0.016 | 0% |
| rules | rules_plus | false_alarm_rate | 2 | +0.000 | 0% |
| rules | knn | cause_macro_f1 | 2 | +0.109 | 100% |
| rules | knn | error_auroc | 2 | -0.014 | 0% |
| rules | knn | false_alarm_rate | 2 | -0.076 | 100% |
| rules | knn_embed | cause_macro_f1 | 2 | +0.119 | 100% |
| rules | knn_embed | error_auroc | 2 | -0.004 | 50% |
| rules | knn_embed | false_alarm_rate | 2 | -0.098 | 100% |
| rules | logreg_checks | cause_macro_f1 | 2 | +0.000 | 0% |
| rules | logreg_checks | error_auroc | 2 | -0.005 | 0% |
| rules | logreg_checks | false_alarm_rate | 2 | +0.000 | 0% |
| rules | logreg_memory | cause_macro_f1 | 2 | -0.039 | 0% |
| rules | logreg_memory | error_auroc | 2 | -0.016 | 0% |
| rules | logreg_memory | false_alarm_rate | 2 | +0.000 | 0% |
| rules_plus | knn | cause_macro_f1 | 2 | +0.167 | 100% |
| rules_plus | knn | error_auroc | 2 | +0.002 | 100% |
| rules_plus | knn | false_alarm_rate | 2 | -0.076 | 100% |
| rules_plus | knn_embed | cause_macro_f1 | 2 | +0.177 | 100% |
| rules_plus | knn_embed | error_auroc | 2 | +0.011 | 100% |
| rules_plus | knn_embed | false_alarm_rate | 2 | -0.098 | 100% |
| rules_plus | logreg_checks | cause_macro_f1 | 2 | +0.058 | 100% |
| rules_plus | logreg_checks | error_auroc | 2 | +0.010 | 100% |
| rules_plus | logreg_checks | false_alarm_rate | 2 | +0.000 | 0% |
| rules_plus | logreg_memory | cause_macro_f1 | 2 | +0.019 | 100% |
| rules_plus | logreg_memory | error_auroc | 2 | +0.000 | 0% |
| rules_plus | logreg_memory | false_alarm_rate | 2 | +0.000 | 0% |
| knn | knn_embed | cause_macro_f1 | 2 | +0.010 | 50% |
| knn | knn_embed | error_auroc | 2 | +0.010 | 100% |
| knn | knn_embed | false_alarm_rate | 2 | -0.022 | 100% |
| knn | logreg_checks | cause_macro_f1 | 2 | -0.109 | 0% |
| knn | logreg_checks | error_auroc | 2 | +0.008 | 100% |
| knn | logreg_checks | false_alarm_rate | 2 | +0.076 | 0% |
| knn | logreg_memory | cause_macro_f1 | 2 | -0.148 | 0% |
| knn | logreg_memory | error_auroc | 2 | -0.002 | 0% |
| knn | logreg_memory | false_alarm_rate | 2 | +0.076 | 0% |
| knn_embed | logreg_checks | cause_macro_f1 | 2 | -0.119 | 0% |
| knn_embed | logreg_checks | error_auroc | 2 | -0.001 | 50% |
| knn_embed | logreg_checks | false_alarm_rate | 2 | +0.098 | 0% |
| knn_embed | logreg_memory | cause_macro_f1 | 2 | -0.158 | 0% |
| knn_embed | logreg_memory | error_auroc | 2 | -0.011 | 0% |
| knn_embed | logreg_memory | false_alarm_rate | 2 | +0.098 | 0% |
| logreg_checks | logreg_memory | cause_macro_f1 | 2 | -0.039 | 0% |
| logreg_checks | logreg_memory | error_auroc | 2 | -0.010 | 0% |
| logreg_checks | logreg_memory | false_alarm_rate | 2 | +0.000 | 0% |

## Caveats

- Laya trains on fewer groups than the MLP and stacking: it holds out a fixed validation set, while they use inner cross-validation on all training groups.
- Laya's banking validation sets have fewer than 20 rows, so Laya keeps temperature 1 (uncalibrated); compare Brier score and ECE with that in mind.
- Each repeat combines the k separately trained fold models.
- Development-scale data: treat these as comparisons, not benchmark claims.
