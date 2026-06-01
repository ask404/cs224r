# ZIP-RC-Lite on Countdown — results
## Calibration (held-out)
| prefix | AUC(value→correct) |
|---|---|
| first | 0.688 |
| q25 | 0.810 |
| mean | 0.904 |
| end | 0.922 |

## Selection @K (held-out)
| K | random | majority | value | oracle |
|---|---|---|---|---|
| 1 | 0.660 | 0.680 | 0.660 | 0.660 |
| 2 | 0.680 | 0.740 | 0.700 | 0.720 |
| 4 | 0.645 | 0.760 | 0.740 | 0.740 |
| 8 | 0.665 | 0.780 | 0.760 | 0.780 |

## Cost–accuracy frontiers (held-out)
| config | acc | oracle | cost | latency |
|---|---|---|---|---|
| none | 0.725 | 0.825 | 4072 | 678.2 |
| prune | 0.675 | 0.700 | 1490 | 528.5 |
| util b=0.02 | 0.625 | 0.625 | 1145 | 465.6 |
| estop t=0.8 | 0.725 | 0.750 | 2500 | 359.5 |

## Adaptive-K allocation (held-out, oracle@meanK)
| mean-K | fixed | hardness | uncertainty |
|---|---|---|---|
| 2 | 0.714 | 0.713 | 0.714 |
| 3 | 0.741 | 0.740 | 0.742 |
| 4 | 0.756 | 0.749 | 0.758 |
| 5 | 0.764 | 0.764 | 0.762 |
| 6 | 0.774 | 0.779 | 0.766 |

