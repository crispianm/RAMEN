# Stage 1 scoring: PLACEHOLDERS (nothing drafted here; the AV2 scorer is separate)
Checkpoints to score: `runs/confirm1_S1_n{46,47,48}_{TS2,H300}/step14000.pt` (and step12500/13000/13500 if the readout wants the trajectory), plus the three start checkpoints `c3_nXX/step12000.pt` as pre-continuation references (already on the cluster; sha in SUBMIT.md).
Slots (to fill when the readout is registered): [ ] AV2 far-GT scorer (confirm/av2/av2_scorer.py, separate task) per run; [ ] DEV guard cells (bench_joint.py DEV, as the screen SCORE_*.sbatch, --split dev, no --allow-test); [ ] DDAD val (report-only, Add.43);
[ ] pairing rule: per pair the contrast is H300 minus TS2 on the SAME start/seed; consistency of sign across the 3 pairs; no pooling rule is set here. The by-range scorer is not needed (Add.48).
Wording (Auditor O39 / T2): "three paired continuations from settled 12k checkpoints (tests recovery, not prevention)".
