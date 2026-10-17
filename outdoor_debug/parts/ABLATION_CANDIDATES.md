# Candidate arms for the pre-declared fine-tune screen (draft, nothing run)
Start: c3_n46 step12000. Light-track rules (Add. O2): log full arm list, N tried / K better, DEV only, no compute before user + Auditor clearance. One change per arm.

| Arm | Change | Extra cost per step |
|---|---|---|
| A0 | continuation baseline (no change) | 0 |
| H80 | one-sided hinge 0.1 @ 80 m on sky | ~0 |
| H300 | same hinge, floor 300 m (80 m equals the training cap, so it barely fires; floor above cap is the real test) | ~0 |
| TS | TartanAir sky-threshold fix (loader rule d>1e4) | 0 |
| T-a | teacher output hinge: log-depth, pred >= teacher, on sky + pixels beyond 80 m only (teacher = frozen released Metric-VDA-S on the CLEAN frames) | +1 frozen VDA-S forward/clip, no grad (~+25-35% step time est., to be measured) |
| T-b | teacher ranking/ordering term on sky + far pixels | same teacher forward + pair sampling (small) |
| T-c | cosine feature distillation at path_3 only | same teacher forward; path_3 features of student already computed, so no extra student cost; cheapest of the teacher arms if the teacher forward is accepted |

Notes
- Teacher already has the target behaviour: KITTI DEV sky 479 m (released) vs 155-191 m ours.
- Caveat: T-a/T-b/T-c inherit the teacher's own far/sky errors and use no real far label; they test "match release", not "match truth". Pair with VKITTI far pred:gt (release 0.995).
- Trajectory result: compression is present by step 6000 and plateaus, so a step-12000 screen tests recovery, not origin. A from-init arm is a later question.
