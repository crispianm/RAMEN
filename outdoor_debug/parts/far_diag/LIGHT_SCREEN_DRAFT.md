# Light screen L1: what makes real far geometry compressed? (DRAFT, not cleared, not submitted)
STATUS: DRAFT for the Manager/Auditor/user. Track 2 (LIGHT, Add. O2; user Add. 46). Requirements of Auditor O34 applied: L-1 NEW sweep (own N, own log section, in-sweep baseline, settings, target + guards, DEV set, decision rule, all declared before any run); L-2 the strict track's held-out set is not touched (no download, look or score); only exposed sets (VKITTI/KITTI DEV, DDAD val) are used and labelled 'exposed'; L-3 A4 handled separately (A4_draft.md, release only). Nothing here is a paper claim, a merge or a retrain proposal.

## Question
Which part of the C3 recipe turns the released Metric-VDA-S's far-range behaviour (DDAD 120-200 m median pred:gt 0.854 at 378; VKITTI 80-655 m 0.995) into the compressed one (0.57 / 0.84) within a short run from the release? FAR_DIAG.md shows: far supervision is 1.4% of SiLog pixels and only synthetic; VDA terms are 5-7x weaker per pixel at far and scale-invariant; compression is already present on trained VKITTI; the 6-arm sky/label screen did not move it. Each arm below removes ONE suspected cause (mechanism = a different reason the far scale could be lost). Origin test, not recovery: every run starts from the RELEASE (T_trajectory could not resolve steps 0-6000), so the screen also checks whether a 2000-step run reproduces the compression (validity gate G0).

## Setting (identical in all runs)
Base config configs/experiments/dn01_c3rep_20261009_c3_n46.yaml (resume null, vda_init released, `data.seed: 46`; `training.seed_scheme: x1000` derives the per-rank seeds from data.seed, so B0b = `data.seed: 49`, the unused seed of the last screen). Overrides common to all: `training.max_steps: 2000`, `training.lr_decay_steps: 2000`, `training.warmup_steps: 100`, `training.severity_warmup_steps: 100`, `checkpoint_at_steps: [2000]`, `training.val_every_steps` large. The recipe (loss, mix, degradation, bf16, 4 GPUs x bs1 x accum4) is otherwise C3. N.B. a 2000-step cosine is a different schedule from the 12k recipe: G0 below guards against a non-informative screen.

## Arms (N tried = 0 now; as declared N = 7 core runs, 5 arms + 2 baselines, counting baselines as the Add. O6b screen did; 8 if S6 is added)
| Arm | Single change vs B0 | Real config knobs | Mechanism tested | Cost |
|---|---|---|---|---|
| B0 | none (in-sweep baseline) | as above | reference | 1 run |
| B0b | data seed 49 (noise sizing only) | `data.seed: 49` | seed spread | 1 run |
| S1 no-indoor | NYU and ScanNet++ removed from the mix | delete the two blocks from `data.sources` (nyu_train, scannetpp_train; 22% of mass; the sampler renormalises) | a 10 m-capped indoor prior pulls the head toward near range | 1 run |
| S2 KITTI/10 | KITTI-raw sampling mass /10 | `data.sources[kitti_raw].sampling_mass: 29940` (from 299397; 27% -> ~3.6%) | a real-image prior 'depth <= 80 m, mostly <55 m' (0 px beyond 80 m, 40x repeats) | 1 run |
| S3 far x3 | VKITTI sampling mass x10 so far SiLog pixels rise ~3x | `data.sources[vkitti].sampling_mass: 400000` (from 40000; VKITTI ~3.6% -> ~30% of clips, ~7 repeats of each clip in 2000 steps); far share of SiLog px 1.4% -> ~4% (computed from Table A: 0.30 x 8.2% + 0.20 x 2.6%) | far supervision is too rare (mix weight) | 1 run |
| S4 lambda | SiLog lambda 0.5 -> 0.15 | `loss.metric_silog_lambda: 0.15` | lambda 0.5 half-forgives a pooled bias, near-over/far-under tilts partly cancel | 1 run |
| S5 head lr | DPT head at 0.1x lr | `training.head_lr_mult: 0.1` (needs `model.freeze_depth_head: false`, already set) | head drift 15-20% from the release erases its far behaviour | 1 run |
| S6 (OPTIONAL, needs a code change; adds 1 to N) far-weighted SiLog | SiLog per-pixel weight 4 for GT >= 50 m | new opt-in keys `loss.metric_silog_far_weight: 4.0`, `loss.metric_silog_far_from: 50.0` in the SiLog block (train.py ~l.1245-1250), default 1.0 = byte-identical: w = 1 + (fw-1)*[depth >= from]; silog = sqrt(sum(w d^2)/sum(w) - lambda*(sum(w d)/sum(w))^2); unit tests: default equals current to 1e-7, gradcheck, far weight raises far pull | pooled-mean weighting under-serves the 1.4% far pixels | 1 run + code diff check |
Not included (reported, not proposed): sky terms and TartanAir sky labels (already screened), 12k-step runs, RAMEN-specific variants.

## Metrics (declared before any run)
- Scoring sets (all EXPOSED, label 'exposed DEV, exploratory'): VKITTI DEV (all scenes are training scenes), KITTI DEV, NYUv2/ScanNet++ DEV guards, DDAD val LR_DEV (50 scenes; exposed by earlier scoring). The strict track's held-out set is not used.
- PRIMARY (training domain): |1 - x|, x = VKITTI 80-655 m legacy_subsample_median pred:gt (scorer.py a8b0731e, same rule as the Add. O6b screen, Auditor O23). Reference = B0 (in-sweep). NF_p = max(|B0 - B0b|, 1% relative).
- INDEPENDENT CHECK (real far GT, exposed, 378 input, scorer 80916c4f): DDAD median pred:gt in [120,200] and [80,120); the arm 'moves toward 1' if |1 - x| shrinks by more than NF_d = max(|B0 - B0b|, 0.01 absolute) in the [120,200] bin and does not grow in [80,120).
- GUARDS: KITTI, NYUv2, ScanNet++ DEV unaligned AbsRel and TAE (tae_gtvalid; KITTI + ScanNet++ only), each not worse than B0 by more than NF; for S1 the NYU/ScanNet++ guards are REPORTED, not gated (the arm removes those sources by design).
- DESCRIPTION ONLY: DDAD overall/near-bin AbsRel, KITTI/VKITTI sky medians, per-bin VKITTI AbsRel, DDAD release row at 518 (A4).
- G0 validity gate: B0 must reproduce the compression: B0 VKITTI 80-655 m legacy median <= 0.93 (release 0.995; the 12k models 0.84-0.86) AND B0 DDAD 120-200 m median <= 0.75 (release 0.854, trained 0.57). If not, the 2000-step screen cannot test causes of compression: stop and report; no arm verdicts.

## Pre-declared rule
Per arm, computed by one script committed before the first score (readout.py pattern):
- CAUSE-IMPLICATED: primary improves by more than NF_p AND the DDAD [120,200] median moves toward 1 by more than NF_d.
- FIX-CANDIDATE: CAUSE-IMPLICATED AND all gated guards within NF.
- PRIMARY-ONLY / DDAD-ONLY: one leg only (read as 'in-domain fit' resp. 'transfer'; stated as such, no cause claimed).
- NOT: otherwise.
A BETTER-type statement here is a screen hit on exposed data. Anything heading for paper, merge or retrain needs a fresh pre-declaration, the Auditor's token and a NEW held-out real far-GT set (not touched here). Nothing is added after any result; a combination of arms is a new sweep with its own N.

## Cost
Training: 2000 steps x 7 runs. Measured basis (SCREEN_PROPOSAL, sacct): ~0.72 h per 1000 steps on 4 GH200 => 1.45 h wall = ~1.45 NHR per run; 7 runs ~10 NHR (S6: +1.45). Scoring (by-range + DDAD per arm, measured in the last screen: 480-500 s + 220-240 s on one GH200): ~0.6 NHR across 7 arms if run as single-GPU jobs ~ 0.2 NHR each; the last screen's total for 6 arms was 9.43 NHR incl. scoring, so ~11 NHR for 7 runs. A4 adds ~0.15 NHR.

## Risks
1. A 2000-step cosine from the release may not reproduce the compression (G0 covers this. DECISION (R+D, Auditor O38 Q2): G0 = STOP AND REPORT. There is no automatic 4000-step fallback; a longer schedule would be a new pre-declaration and a new sweep. Read G0 from B0 and B0b before any arm is scored/read).
2. One baseline pair understates seed variance (the last screen's A0/A0b spread on NYUv2 was 0.097 vs 0.077); an arm effect below NF is not an effect.
3. VKITTI is a training set: the primary measures fit, not generalisation; DDAD is exposed and a single real set: a hit may be DDAD-specific (camera/FOV).
4. S1 and S2 change the mix, so they also change the near-range anchor; a far gain with a near loss is a trade, read with the guards.
5. S3 raises VKITTI repeats to ~7 per clip in 2000 steps (memorisation risk, visible as a primary gain without a DDAD gain).
6. Several knobs (lambda, head lr) interact with the short schedule; a negative result does not rule them out at 12k steps.

## Held-out far-GT candidates (report only; NOT acquired, NOT touched; L-2)
(a) DDAD train split (150 scenes of the same sensor, different scenes from val; same 275 GB tar, no sha published; would need a re-download and its own registration); (b) [struck: the strict track's held-out set; Track 2 may not name, list, acquire or score it (Auditor O38 L-2)]; (c) Waymo Open / nuScenes are limited to ~75 m / ~50-70 m LiDAR and do not cover >80 m well. None is used by this draft; any would need its own registration and the Manager's approval.


## Auditor O38 additions (draft, R+D)
- Schedule declared numerically for every run: max_steps 2000, warmup 500 steps (linear), cosine to 0 at step 2000; no resume. First-run LR stop check against the closed form (as the confirm spec: compare every printed lr to 1e-4 x scale), with negative controls. B0 and B0b run FIRST and G0 is read before the arms run.
- G0 vs the fallback: G0 fails => STOP and report (no arm verdicts, no 4000-step stage).
- B0b seed: no resume, so data.seed 49 through the x1000 scheme is sufficient; a batch proof is required before the run: B0 vs B0b first-N clip ids and a noise checksum DIFFER. S1/S2/S3 change the data mix, so their clip ids legitimately differ from B0's.
- DDAD as a decision leg: only with the user's explicit word (it changes Add. 43's report-only); if allowed it is labelled 'exposed DEV gate' and exploratory only. Default in this draft: DDAD is report-only.
- S6 only if declared before the first run (otherwise not run).
- A4 conditions: A4 filed in REGISTRATION.md first; the patch committed with a new scorer sha; tests including the refusal paths; the first 518 run's shape log; every comparison names both input sizes (378 for ours, 518 for the release row).
