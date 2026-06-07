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

## Cost–accuracy frontiers (held-out)
| config | acc | oracle | cost | latency |
|---|---|---|---|---|
| none | 0.708 | 0.812 | 4141 | 688.4 |
| prune | 0.708 | 0.729 | 1266 | 497.5 |
| util b=0.005 | 0.708 | 0.729 | 3105 | 519.4 |
| util b=0.01 | 0.688 | 0.708 | 2343 | 488.7 |
| util b=0.02 | 0.667 | 0.688 | 1641 | 489.3 |
| util b=0.05 | 0.625 | 0.625 | 1160 | 481.5 |
| estop t=0.7 | 0.688 | 0.708 | 2504 | 389.2 |
| estop t=0.8 | 0.729 | 0.750 | 2645 | 384.2 |
| estop t=0.9 | 0.729 | 0.750 | 3161 | 439.8 |
| estop t=0.95 | 0.729 | 0.771 | 3586 | 537.3 |

## Adaptive-K allocation (held-out, oracle@meanK)
| mean-K | fixed | hardness | uncertainty |
|---|---|---|---|
| 2 | 0.714 | 0.732 | 0.719 |
| 3 | 0.741 | 0.748 | 0.735 |
| 4 | 0.756 | 0.765 | 0.749 |
| 5 | 0.764 | 0.778 | 0.758 |
| 6 | 0.774 | 0.780 | 0.762 |

