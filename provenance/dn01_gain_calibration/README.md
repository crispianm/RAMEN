# DN-01 v2 blind display gain: calibration (training clips only, before any v2 job)

Auditor K43(b): a BLIND per-clip global gain computed from the INPUT only, its formula fixed before any job, the same
function for ST and every cascade. `bench_joint.blind_gain(x) = clip(GAIN_REF / stat(x), 1, GAIN_MAX)`, with
`stat = bench_joint.gain_stat(x, GAIN_STAT)` = the median over frames of each frame's luma statistic.

Run on crispwork, 2026-09-26, by `scripts/dn01_gain_calibrate.py --config configs/experiments/dn01_v2_ll_ctrl.yaml`
(the real train loader, local VKITTI/IRS roots, the v2 C-LL mix: 50/50 awgn_blur | lowlight, EV U[-3, -0.7],
kappa 0.186, clean_prob 0.2, blur_only_prob 0.5, severity cap 1.0; 8-frame clips). No eval clip was read.

- `gain_calibration.json`: 400 FIT clips + 200 disjoint CHECK clips (the loader's own order; the mix is IRS-heavy,
  as the hunt's probe found: 236 / 240 IRS). Candidates: luma mean, q50, q90, q95, q99, q99.5; GAIN_REF grid
  2^linspace(-4, 1, 1001); the lowest fit MAE wins.
  - **Chosen: GAIN_STAT = q99.5, GAIN_REF = 0.8796490759224356, GAIN_MAX = 16** (fit MAE 0.202 stops; q99 0.216,
    q95 0.265, q90 0.313, mean 0.431, q50 0.581).
  - CHECK (200 clips, 84 darkened): MAE 0.216 stops, median 0.023; darkened 0.269, normally exposed 0.179; 9.5% of
    normally exposed clips are brightened by > 0.5 stop (dark indoor scenes; the gain never goes below 1).
- `vkitti_only/gain_calibration.json`: `--only-source vkitti` (the outdoor source; the local copy is the Scene01
  clone only, so the loader ends after 56 clips). This is a per-source ERROR REPORT, not a refit; the chosen rule
  was not changed. The chosen rule on all 56 VKITTI clips: MAE 0.020 stops, median 0.000; darkened (17) 0.058;
  normally exposed 0.004; none brightened by > 0.5 stop; max 0.133.

Code state of the main run: base commit 9d3bef2 + `run_state.patch` (sha256 c6a87e85..., = `git diff HEAD` taken
right after launch) + the two then-untracked files `run_state_dn01_v2_ll_ctrl.yaml` (sha256 47457508...) and
`run_state_dn01_gain_calibrate.py` (sha256 93dccfbe..., = the JSON's `script_sha256`). The JSON's own
`uncommitted_diff_sha256` was taken at the END of the run, after unrelated edits (tests, bench_joint's scoring
block) had been made meanwhile, so it does not match `run_state.patch`; the patch is the authoritative state.
bench_joint's GAIN_* were None during the run (gain_stat does not read them). The VKITTI report ran at the same
base with `--only-source` added to the script (its JSON's `script_sha256`).
