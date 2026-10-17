## Table A: per source, % of ALL pixels of the sampled clips (8-frame clips at 518, 40 clips/source, seed 45)

| source | mass share | clips | sky % | valid % | 0-20 | 20-50 | 50-80 | 80-120 | 120-200 | 200-655 | >=655 | SiLog-eligible % (all bins) | SiLog-elig >=80 m % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tartanair | 27.9% | 40 | 7.00 | 93.0 | 75.600 | 8.172 | 1.725 | 1.092 | 0.753 | 0.723 | 4.927 | 88.1 | 2.568 |
| vkitti | 3.6% | 40 | 17.53 | 82.5 | 49.235 | 19.838 | 5.185 | 3.150 | 3.260 | 1.799 | 0.000 | 82.5 | 8.209 |
| pointodyssey_train | 7.9% | 40 | 0.00 | 86.3 | 83.981 | 2.168 | 0.192 | 0.008 | 0.000 | 0.000 | 0.000 | 86.3 | 0.008 |
| pointodyssey_val | 1.2% | 40 | 0.00 | 88.8 | 88.021 | 0.816 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 88.8 | 0.000 |
| irs | 9.0% | 40 | 0.00 | 97.5 | 97.351 | 0.163 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 97.5 | 0.000 |
| spring | 0.4% | 40 | 0.00 | 87.6 | 67.243 | 17.966 | 2.395 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 |
| kitti_raw | 27.0% | 40 | 0.00 | 17.3 | 13.228 | 3.573 | 0.503 | 0.000 | 0.000 | 0.000 | 0.000 | 17.3 | 0.000 |
| nyu_train | 15.3% | 40 | 0.00 | 96.4 | 96.354 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 96.4 | 0.000 |
| scannetpp_train | 7.7% | 40 | 0.00 | 99.9 | 99.935 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 99.9 | 0.000 |

## Table B: mass-weighted MIX (expected fraction of all pixels in a sampled batch; weight = mass_s / sum mass)

| quantity | 0-20 | 20-50 | 50-80 | 80-120 | 120-200 | 200-655 | >=655 | total |
|---|---|---|---|---|---|---|---|---|
| valid GT px (VDA loss mask) % | 65.599 | 4.235 | 0.830 | 0.419 | 0.328 | 0.267 | 1.376 | 73.06 |
| SiLog-eligible px % | 65.306 | 4.157 | 0.820 | 0.419 | 0.328 | 0.267 | 0.000 | 71.30 |
| share of SiLog-eligible px | 91.60 | 5.83 | 1.15 | 0.59 | 0.46 | 0.37 | 0.00 | 100 |

sky px (VDA zero-disparity target only) = 2.59% of all px

## Table C: far (>=80 m) SiLog pixel share of the mix by source

- tartanair: 0.718% of all px = 70.8% of far SiLog px;  120-655 m: 0.412%
- vkitti: 0.296% of all px = 29.2% of far SiLog px;  120-655 m: 0.182%
- pointodyssey_train: 0.001% of all px = 0.1% of far SiLog px;  120-655 m: 0.000%
- pointodyssey_val: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- irs: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- spring: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- kitti_raw: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- nyu_train: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- scannetpp_train: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- mix total >=80 m SiLog-eligible: 1.014% of all px = 1.42% of SiLog px
