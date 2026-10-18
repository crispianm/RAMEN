# Overnight report 2026-10-18 (R+D [e52384]); everything below is exploratory, exposed DEV data, unaudited unless a ruling is named; no fix, paper or merge claim.
Reading guards: partly by construction (VKITTI is the same far distribution that the VKITTI probe measures); one baseline pair (B0/B0b) understates seed variance; release-model comparisons only at equal input size (release at its registered 378 beside ours at 378; the release at 518 beside the release at 378, A4 row, never against ours); debug only, not paper.

### L1 (K=2000) eight-arm table: exploratory, exposed DEV, unaudited

Noise floors from |B0-B0b| (dev = |1-x|): NF_p (VKITTI) = 0.03229; NF_d (DDAD [120,200]) = 0.02604; NF_b (DDAD [80,120)) = 0.01518.

| arm | verdict | x_V | x_D[120,200] | dev_D[80,120) | DDAD overall AbsRel | kitti.absrel (NF 0.0056) | nyuv2.absrel (NF 0.0020) | scannetpp.absrel (NF 0.0022) | kitti.tae (NF 0.0200) | scannetpp.tae (NF 0.0197) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | 0.8605 | 0.6428 | 0.2298 | 0.1909 | 0.1288 | 0.0807 | 0.1496 | 1.9973 | 1.9678 |
| B0b | baseline | 0.8928 | 0.6167 | 0.2450 | 0.1906 | 0.1343 (+0.0056) | 0.0827 (+0.0020) | 0.1474 (-0.0022) | 1.9835 (-0.0138) | 1.9781 (+0.0103) |
| S1 | NOT | 0.8458 | 0.6375 | 0.2235 | 0.1924 | 0.1321 (+0.0034) | 0.1126 (+0.0319) | 0.2172 (+0.0677) | 2.0335 (+0.0362) | 2.2732 (+0.3054) |
| S2 | DDAD-ONLY | 0.8822 | 0.7974 | 0.1186 | 0.2040 | 0.1246 (-0.0041) | 0.0791 (-0.0016) | 0.1355 (-0.0141) | 1.9287 (-0.0686) | 1.9283 (-0.0395) |
| S3 | FIX-CANDIDATE | 0.9527 | 0.6891 | 0.1904 | 0.1906 | 0.1317 (+0.0029) | 0.0781 (-0.0025) | 0.1491 (-0.0005) | 1.9834 (-0.0139) | 1.9847 (+0.0169) |
| S4 | NOT | 0.8654 | 0.6427 | 0.2324 | 0.1921 | 0.1295 (+0.0007) | 0.0822 (+0.0015) | 0.1472 (-0.0024) | 2.0082 (+0.0109) | 1.9852 (+0.0174) |
| S5 | CAUSE-IMPLICATED | 0.9000 | 0.6702 | 0.2085 | 0.1935 | 0.1342 (+0.0054) | 0.0802 (-0.0004) | 0.1574 (+0.0078) | 2.0118 (+0.0145) | 2.0590 (+0.0912) |
| S6 | NOT | 0.8741 | 0.6014 | 0.2331 | 0.2007 | 0.1342 (+0.0055) | 0.0809 (+0.0002) | 0.1493 (-0.0003) | 2.0212 (+0.0239) | 1.9691 (+0.0012) |

Guard cells show the value and (change vs B0). A guard is 'worse than NF' when the change exceeds its NF (KITTI/NYUv2/ScanNet++ abs_rel_unaligned, KITTI/ScanNet++ tae_gtvalid; S1's NYUv2/ScanNet++ guards are reported-only by design).

### AV2 Stage 1 (confirmatory strict, TS2+H300 vs TS2, three pairs): verdict NOT CONFIRMED

| pair | P_H300 (m) | P_TS2 (m) | d = ln(P_H300/P_TS2) |
|---|---|---|---|
| s50 | 444.6 | 265.7 | 0.5149 |
| s51 | 449.3 | 263.1 | 0.5352 |
| s52 | 426.1 | 239.2 | 0.5772 |

d_mean 0.5425, d_min 0.5149, ln 1.5 = 0.4055. Guards (H300 AbsRel / TS2 AbsRel, limit 1.05; px per bin 3.09M / 3.92M / 1.33M / 0.66M / 0.36M, none unevaluable):

| pair | [0,20) | [20,50) | [50,80) | [80,120) | [120,200] |
|---|---|---|---|---|---|
| s50 | 0.9809 | 0.9965 | 0.9941 | 0.9953 | 0.9998 |
| s51 | 1.0046 | 1.0169 | 1.0248 | 1.0284 | 0.9722 |
| s52 | 0.9959 | 1.0298 | 1.0521 **FAIL** | 1.0395 | 0.9724 |

Verdict string and disclosure (verbatim): NOT CONFIRMED | Stage 1: THREE paired 2000-step continuations (n46/s50, n47/s51, n48/s52); 3 pairs size the pair-to-pair spread only roughly; the A0/A0b screen pair showed 26% spread on a guard. P measures what the H300 hinge enforces by design (a PASS shows transfer of the hinge's effect, not accuracy; the guards carry the information). TS2+H300 was chosen after the screen on a descriptive sky secondary and was NOT BETTER on its primary (the screen primary).

Reference row (description only; released Metric-VDA-S at the registered 378 input through the same scorer): P (sky median) 582.8 m; per-bin AbsRel (pixel-pooled, unaligned): [0,20) 1.256, [20,50) 0.970, [50,80) 0.716, [80,120) 0.581, [120,200] 0.441; median pred:gt: [0,20) 2.118, [20,50) 1.897, [50,80) 1.553, [80,120) 1.280, [120,200] 0.978. (The AV2 portrait camera makes focal length ambiguous: the high AbsRel is equal across arms.)

### Final NHR (sacct elapsed node-hours since 2026-10-17T16:00Z; total 34.33; cap 45)
| block | NHR |
|---|---|
| data/audit jobs before the screen (DDAD extract, TartanAir sky audit) | 0.13 |
| screen (earlier): smokes 0.12 + training 8.93 + by-range scoring 0.96 + DDAD scoring 0.56 | 10.57 |
| srun optimizer probes (5) | 0.02 |
| A4 (release at 518): failed run 1 (checker defect) + rerun | 0.13 |
| L1 smokes S1/S5/S6 | 0.34 |
| L1 baselines B0/B0b training | 2.99 |
| L1 baseline scoring: two failed starts (pin mismatch, ~0) + rescoring | 0.39 |
| L1 score-pin identity control | 0.06 |
| L1 arms S1-S6 training | 8.74 |
| L1 arms S1-S6 scoring | 1.17 |
| Stage 1 smokes | 0.12 |
| Stage 1 six fulls | 9.28 |
| AV2 Stage 1 scoring (6 arms + reference) | 0.40 |
| total | 34.33 |
Failed starts: the L1 baseline score jobs 7511130/7511131 (failed closed on a scorer-sha pin mismatch, ~0 NHR); A4 run 1 (job 7509896, 3:59, checker defect, 0.066 NHR, output sealed and superseded).

### Committed files and shas (branch outdoor-debug)
- L1 readout (sealed file, now cleared): outdoor_debug/parts/far_diag/l1_final/scores_k2000_sealed/L1_READOUT_SEALED.txt, sha256 5c7600a875c8f6738749f199dd597e5e941b87be875a765830861320d58913a4; all eight score dirs in the same folder; G0 readout: g0_stage2000/g0_readout_output.txt 2900ad14…
- Stage 1 AV2: verdict JSON (Auditor-verified byte-identical) in the R+D job tmp av2_stage1_verdict_SEALED.json sha 3bac237c9e18cca7b52a0665d570a23a94149996d870cf5fef080ed3d40f996f (copied into the repo with the mutant table commit), stdout a258c086…; six run jsons + reference row in outdoor_debug/parts/confirm/av2/stage1_runs_sealed/ with av2_stage1_run_shas.txt; mask panel outdoor_debug/parts/confirm/av2/mask_build/mask_panel.png sha 49e4df48b4d350b9edac35520df9d11cfc634f37ccf07837f5aaf78f68b975b8; mask npz de2e9ca4177691ee6e3b904481e4177346192afdf617f24eb2c147c93258f1b1 (workstation + Isambard, read-only, not committed), meta 10285355e791c9e9032ecb5b00d51c412065e7741b698f0f19a493966e4fa073.
- L1 identity control (score-pin 8e71511b vs ddad_ref2, max diff 0.000e+00): outdoor_debug/parts/far_diag/l1_final/identity_control/identity_control_result.txt sha bb083c25…
- Registrations/records: AV2_REGISTRATION.md (R1-R4), ABLATION_CANDIDATES.md (job ids logged before reads).

### Still running / uncommitted at the time of writing
- The AV2 and L1 mutant re-run on the filled pins is still running locally (about 100 mutants; first run aborted at one stale mutant, 71 killed and 0 survived until then; restarted); the table is committed when it finishes and sent to the Auditor.
- Nothing is running on Isambard. The AV2 Stage 2 subset (b5b94e1b) is untouched and Stage 2 did not run.
- Not committed to the repo: the verdict JSON copy (committed with the mutant table), the local checkpoints of the six Stage 1 runs (workstation: /mnt/DATA/DegradedDepth/checkpoints_v3/confirm1_stage1_20261018; cluster: /projects/b5dh/RAMEN_confirm1_f8ac6d0b/runs).
