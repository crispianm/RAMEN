# Screen scoring jobs (by-range scorer) -- DRAFT, NOT SUBMITTED

Supersedes ../score_template.sbatch / ../SCORE_*.sbatch (they ran bench_joint only). 7 single-GH200 jobs: REF (c3_n46 step12000) + 6 arms (A0 A0b TS2 TS2_H80 TS2_H300 TS2_VK655).

## Facts verified (read-only ssh, 2026-10-17 ~18:40Z)
- Trainer writes `step{N}.pt` for N in `training.checkpoint_at_steps` (all 6 YAMLs: 12500,13000,13500,14000; train.py l.2979: inside the val/save block) plus `last.pt`, `final.pt` (end of run) and `best.pt`. Arm dirs now hold `last.pt`, `step12500.pt` (runs RUNNING, 7504894/97/98/900/902/904). The score jobs use `runs/outdoor_screen_<ARM>/step14000.pt` and check_scores asserts `meta.ckpt.step == 14000` (so a wrong file cannot pass). Not yet present, so existence is checked at job start.
- bench_joint needs no path override: `PATHS = _HOSTS["isambard" if Path("/projects/b5dh").is_dir()]`; evaluation/vda5_sets.py has its own isambard table. All present: kitti_rgbfix + kitti_video.json, dn01_eval_data/kitti_raw_val_drives (all 7 DEV zips: 2011_09_28_drive_0037, 2011_09_26_drive_0005/0023/0020/0113/0095, 2011_10_03_drive_0047), dn01_eval_data/kitti_calib/{2011_09_26,28,29,30,10_03}, data/test/benchmarks/nyuv2_video/nyuv2_video_500.json, scannetpp_eval/scannetpp_video.json, data/train/vkitti (Scene*, e.g. Scene20/overcast), repos/Video-Depth-Anything, venv /projects/b5dh/FocusMamba/.venv/bin/activate. Nothing missing. Not checked file-by-file: the ScanNet++ per-clip frames and VKITTI depth PNGs (the job fails loudly if absent). No horse.
- The scorer commit is NOT in the training clone (2a72223, origin https://github.com/crispianm/RAMEN.git). So a SECOND clone is used.

## Second clone (login node, once, after the parent commits scorer + this dir)
```
export SCORE_COMMIT=<40-hex sha of the commit containing scorer.py (a8b0731e..), bench_joint.py (5d3d048c..) and outdoor_debug/parts/submit/score/*>
git clone --branch outdoor-debug https://github.com/crispianm/RAMEN.git /projects/b5dh/RAMEN_outdoor_screen_scorer   # or bundle relay as in ../submit.sh Option B (GitHub reachability from the cluster is UNVERIFIED)
cd /projects/b5dh/RAMEN_outdoor_screen_scorer && git checkout --detach "$SCORE_COMMIT" && test "$(git rev-parse HEAD)" = "$SCORE_COMMIT" && test -z "$(git status --porcelain)" && mkdir -p logs
```
`logs/` and `runs/` are gitignored, so the tree stays clean. The tree guard in the job also aborts on any dirty file. The outdoor-debug branch must be pushed (or bundled) by the user/parent: this subagent did not commit or push.

## Sha injection (no commit loop)
The committed `SCORE_*.sbatch` hold `SCORE_COMMIT="UNSET"` (a job from them aborts). The sha is written into COPIES outside the clone (gen_score.sh writes beside itself, so it is run from a scratch copy; tested):
```
cd /projects/b5dh/RAMEN_outdoor_screen_scorer
D=/projects/b5dh/RAMEN_outdoor_screen_scores_sbatch_$SCORE_COMMIT; mkdir "$D"
cp outdoor_debug/parts/submit/score/{gen_score.sh,score_template.sbatch} "$D"/ && bash "$D"/gen_score.sh $SCORE_COMMIT
grep -h '^export SCORE_COMMIT' "$D"/SCORE_*.sbatch | sort | uniq -c     # must show 7 x the same sha
```
The files in `$D` are what is submitted (the job still checks HEAD == that sha, clean tree, and both file shas).

## Submit commands (parent runs on approval; reference row first = the S3 cross-host row; arms afterok on their training job)
```
cd /projects/b5dh/RAMEN_outdoor_screen_scorer; D=/projects/b5dh/RAMEN_outdoor_screen_scores_sbatch_$SCORE_COMMIT
R=$(sbatch --parsable $D/SCORE_REF_step12000.sbatch); echo REF $R
# train job ids (sacct 18:40Z): A0 7504894, A0b 7504897, TS2 7504898, TS2_H80 7504900, TS2_H300 7504902, TS2_VK655 7504904
for p in A0:7504894 A0b:7504897 TS2:7504898 TS2_H80:7504900 TS2_H300:7504902 TS2_VK655:7504904; do
  A=${p%%:*}; J=${p##*:}; sbatch --parsable --dependency=afterok:$J $D/SCORE_$A.sbatch
done
```
`#SBATCH --output=logs/%x_%j.out` is relative: submit from the clone dir (logs/ exists). Submitting the arms with `--dependency=afterok` is the only way a missing step14000.pt would not even start; if a training job fails, its score job is held DependencyNeverSatisfied (scancel it).

## What each job does / guards
1. `SCORE_COMMIT` must be 40-hex (placeholder `UNSET` aborts), `git rev-parse HEAD == SCORE_COMMIT`, clean tree, scorer.py sha a8b0731e.., bench_joint.py sha 5d3d048c.., check_scores.py present, checkpoint exists, ckpt sha printed (REF: asserted f5c7c8f9..), output dir `/projects/b5dh/RAMEN_outdoor_screen_scores/<LABEL>/` must not exist (labels: REF_step12000, A0, A0b, TS2, TS2_H80, TS2_H300, TS2_VK655). All variables exported or literal; commit printed in the log.
2. `python -I scorer.py --ckpt $CKPT --out $OUT/scores.json --datasets kitti,nyuv2,scannetpp,vkitti --device cuda --precision fp16 --nyuv2-frames 160` = run_regression.sh flags minus horse; log tee'd to `scorer.log`.
3. `check_scores.py scores.json --expect-step N [--expect-ckpt-sha] --expect-host-contains GH200` (27 checks: pins_asserted, scorer/bj sha, combined clip sha, weights_key, step, ckpt sha, GH200 string in meta.host|gpu, nyuv2_frames 160, fp16, no limit flags, 4 datasets with 7/4/31/10 clips, finite AbsRel, finite tae+tae_gtvalid for kitti and scannetpp incl. every clip, primary far statistic). Any FAIL = job exit 1 and the dir is renamed `<LABEL>.FAILED_<jobid>` (never silently kept). Output in `check_scores.txt`.
4. Copies HEAD.txt, checkpoint_sha256.txt, scores.json, scorer.log, check_scores.txt, (partial) slurm stdout and SHA256SUMS to the score dir. (The final slurm .err/.out stay in clone `logs/`.)

## Open items
- Time limit 6 h is a placeholder; tighten to ~1.5 x the first measured run (REF; the 4090 took 755 s) before the 6 arm rows.
- The scorer `meta.gpu` string on GH200 is assumed to contain "GH200" (nvidia-smi name "NVIDIA GH200 480GB"); if the REF job fails only that check, loosen `--expect-host-contains` to `NVIDIA` and record the real string (the scores.json is kept in the .FAILED dir).
- Cross-host tolerance REF(GH200) vs the 4090 regression row: 1% / 2% as in the scorer README (compare_regression.py not wired into the job; run it from the workstation on the pulled scores.json).
- Not scored here: horse, DDAD, TartanAir far.
