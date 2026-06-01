# ZIP-RC-Lite on Countdown — results
## Calibration (held-out)
| prefix | AUC(value→correct) |
|---|---|
| first | 0.849 |
| q25 | 0.805 |
| mean | 0.899 |
| end | 0.910 |

## Selection @K (held-out)
| K | random | majority | value | oracle |
|---|---|---|---|---|
| 1 | 0.660 | 0.680 | 0.660 | 0.660 |
| 2 | 0.680 | 0.740 | 0.700 | 0.720 |
| 4 | 0.645 | 0.760 | 0.720 | 0.740 |
| 8 | 0.665 | 0.780 | 0.740 | 0.780 |

## Cost–accuracy Pareto (held-out)
| config | acc | oracle | cost | latency |
|---|---|---|---|---|
| none | 0.700 | 0.825 | 4072 | 678.2 |
| prune | 0.700 | 0.725 | 1250 | 480.9 |
| util b=0.002 | 0.700 | 0.750 | 3299 | 512.5 |
| util b=0.005 | 0.700 | 0.725 | 3084 | 497.6 |
| util b=0.01 | 0.675 | 0.700 | 2429 | 475.1 |
| util b=0.02 | 0.650 | 0.675 | 1703 | 470.2 |
| util b=0.05 | 0.600 | 0.600 | 1152 | 472.8 |
| util b=0.1 | 0.600 | 0.600 | 1128 | 448.8 |

