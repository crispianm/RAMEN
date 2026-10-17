# R3 pin: bench_joint commit + DEV clip-list sha (logged 2026-10-17T17:40:16Z, before any training run)
- repo HEAD when computed: 49f13dc2099ab59f2e1d328e084b767ecbff9e3e (branch outdoor-debug); bench_joint.py sha256 5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef. bench_joint.py is unmodified by this task (last changed in 5a2db28, the repo root commit).
- DEV clip lists from bj.select_clips(ds, "dev") keys (one key per line, in returned order):
| dataset | n clips | sha256 of keys |
|---|---|---|
| kitti | 7 | e187a28b87d43f3b3e6f687a0acc944f216fa754b4512fc373d616c5bece0a3a |
| nyuv2 | 4 | 49adc06914d2bfcdec0d602a37099f0289e498780302df39f512a010c6e58797 |
| scannetpp | 31 | 010a310b828dedeaac77700463d243644d83b35a8146d75fa88da79862bb37c6 |
| vkitti | 10 | 2f57ab4872f2ff8e82e86c30f89fa21de5db3351d457df7dc9bb4e7389f15073 |
- COMBINED sha256 over lines "ds:key" (kitti, nyuv2, scannetpp, vkitti in that order): aca96bbd894855cc99b8d97dbdf6e957d1c1f4efb0d49c77ff6108d8dccd7de6
- TartanAir far / horse / KITTI-sky secondary metrics and DDAD (LR_DEV, own registration, report-only) are not in bj.DATASETS here; their frame lists are pinned separately (DDAD: ddad_val_frames.tsv b8d9664f...).
- scorer short_side for DDAD is still provisional (378); to be fixed before the first DDAD score.
