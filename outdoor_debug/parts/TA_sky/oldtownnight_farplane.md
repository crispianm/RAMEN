# OldTownNight far-plane check (TRAIN data, no model output)

Data: oldtownnight_stats.csv (27 frames: all 3 trajectories x q=0.1..0.9, depth only). Panels: /mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_visual_unlabelled_ext/OldTownNight.png (q=0.2/0.5/0.8 of P000; RGB | seg-sky (none) | saturated (none) | log10 depth).

Findings
- No saturated or non-finite px in any of the 27 frames. Per-frame finite max ranges 10.6 .. 231.4 m (sorted: 10.6, 16.3, 24.8, 48.1, 90.9 ... 173, 215, 218.3, 218.5, 219.3, 230, 231.4); no common far-plane value.
- No plateau at the maximum: px with d == max are 1-30 per frame; the top values descend continuously (e.g. P000 q0.5: 152.88x1, 152.75x7, 152.62x11, 152.50x15, 152.38x20; q0.8: 218.25x1, 218.12x6 ...). px >= 0.98*max are 1 .. 12852 per frame (13 frames < 100 px).
- Visually (fixed frames, P000 q0.5 and q0.8; q0.2 is a black frame with max 10.6 m): the cloudy sky is a smooth depth ramp (teal in the depth panel) of ~100-230 m that is of the same magnitude as far buildings/trees; the sky depth is not separable from far geometry by a threshold, and any per-frame 'max' rule would flag the farthest geometry in sky-free frames (q0.2, q0.9 of P000: max 10.6 m / 48 m).

Conclusion: NOT a clean plateau; a far-plane rule d >= F cannot be declared (no fixed F, no plateau, not separable). Reported as a RESIDUAL, not a rule (Auditor O12.5): OldTownNight keeps the released behaviour (sky px carry valid metric depth 10-230 m, sky=0, mask=1; guard and TS v2 do not touch them). No test added (no rule).
