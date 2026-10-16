#!/bin/bash
# PURE variables (no side effects), sourced by every s3_* script and runnable anywhere (DRY submit). Lives OUTSIDE the code clones (copied to $S3_ROOT/scripts). Mechanical adaptation of r1_vars.sh / s0_vars.sh (R1 package, Addendum N version).
# Nothing here writes into any clone, the venv, the eval clone or the VDA repo.
# ===== LADDER code clone (generator): commit I3 = a child of commit I' (3a4884cc, branch paper-repro-Ip) that changes ONLY configs/paper_results.yaml (union smoke: block, tab:noise without the llb3kappa1/st_llb3kappa1 pair) + its README + the freeze tests (ladder_I3 package; Auditor Addendum 7). Pushed as origin paper-repro-I3. =====
S3_PARENT_I=3a4884cce837405589283081d33d222b3cc032ed                                           # commit I' (origin paper-repro-Ip; descends from I 8a99f9b3); S3_WANT must have it as its (first) parent
S3_WANT=1a67278bc331be59404979f4ab2e3bcd1ca69655                                           # I3 (origin paper-repro-I3); the code clone must be CLEAN at exactly this commit
S3_CONFIG_SHA16=c27739c4194afa48                                                               # first 16 hex of sha256(configs/paper_results.yaml) at I3; full c27739c4194afa48a2b5f01e98ea50ce76a386b645f5f973966dd12589209d7b; at I' it is 73aa5ee068176a13, at I eafa3e62ace87648
S3_GEN_SHA16=92bd4e124430c143                                                                  # sha256(generate_results.py) first 16 at H a1493364 = I = I' (blob d4fa03d8): must be UNCHANGED at I3
# ===== FILTER-BED driver clone: the ORIGINAL FBASE driver, commit 9f14d1e (branch v3-fbase-grid, parent = scorer 368b5c53, touches only scripts/dn01_headline.py + its test) =====
S3_DWANT=9f14d1e8fedc680f4fab3fb1067434c318ef17d8
S3_DBRANCH=v3-fbase-grid
# scorer files that must be BYTE-IDENTICAL in driver clone (9f14d1e), code clone (I3) and scorer clone (368b5c53) = H/I (git blob / tree ids, checked 2026-10-15 in the main repo)
S3_BLOBS="bench_joint.py=16fa6d7e6162916abc9b75a6b89020e1beb2d665 metrics.py=2a76eac8fbc87bdb82adc143b13f8072db397ddc metrics_temporal.py=74d9fa56d75153dd30985ceaf157b51fb6f27a55 evaluation=68d9ebd7e9b08e2c5443778f0725254f9dee447e"
# ===== paths / constants =====
: "${S3_ROOT:=/projects/b5dh/RAMEN_v3_results/S3_FT_20261015}"
: "${S3_CODE:=/projects/b5dh/RAMEN_I3_S3_20261015}"                           # fresh clone of paper-repro-I3 (git clone --branch <branch>; checkout --detach $S3_WANT; clean). Created by s3_prep_login.sh; NOT an existing clone.
: "${S3_DCODE:=/projects/b5dh/RAMEN_S3F_DRV_9f14d1e_20261015}"                # fresh clone of v3-fbase-grid at 9f14d1e (the old /projects/b5dh/RAMEN_fbgrid_inv is NOT reused). Created by s3_prep_login.sh.
: "${S3_SCORER_CLONE:=/projects/b5dh/RAMEN_eval_v3_EFF_20261014}"            # scorer clone @368b5c53 (read-only; re-verified 2026-10-15 over ssh: HEAD 368b5c53, 0 porcelain lines)
: "${S3_VDA_REPO:=/projects/b5dh/repos/Video-Depth-Anything}"
: "${S3_VENV:=/projects/b5dh/FocusMamba/.venv}"
: "${S3_BVI:=/projects/b5dh/data/train/bvi_rlv/BVI-Lowlight-videos}"
S3_SCORER_COMMIT=368b5c53b37ebd22f5d44ce0c9d514122efc88da
S3_TOKEN=AUD-FTONLY-S3                                                         # token name per the Auditor scoping Addendum 1 (cap 9.5 NHR); the Auditor's written issue may add a suffix: then edit here
S3_VDA_DIFF16=a1f7fb70c5f905c6; S3_VDA_STATUS16=874bb020                      # BEFORE-record of the VDA repo (same as R1/S0); diff sha re-read over ssh 2026-10-15 = a1f7fb70c5f905c6
# ===== checkpoint pins (sha256), re-verified READ-ONLY on Isambard 2026-10-15 (sha256sum over ssh); ladder job hashes all 6, filter job hashes FT n46 =====
S3_RAMEN_N46_PATH=/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt; S3_RAMEN_N46_SHA=f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af
S3_RAMEN_N47_PATH=/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt; S3_RAMEN_N47_SHA=2ee98b7ee25bd6d5f044ea1f6bb7854d1a0c978488be52d4c87d305a5df6a54c
S3_RAMEN_N48_PATH=/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt; S3_RAMEN_N48_SHA=bf053bd38fdb97d55228fceb737fc2d6353cca26d9ab13338c6d9dd3df95da12
S3_FT_N46_PATH=/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt;   S3_FT_N46_SHA=18c9d8caeab1ab88d52b92782e307bcb8d82897e8a8017aed172f06c26af92a0
S3_FT_N47_PATH=/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n47/step12000.pt;   S3_FT_N47_SHA=ec38c3179cb63349d5642ffae0dc8acecf79ca4ed7525e9dd6a5735eb62cc465
S3_FT_N48_PATH=/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n48/step12000.pt;   S3_FT_N48_SHA=d769e513c6e8995c53f021bf0335a4d040005e7bcadbef1bf8c6d4c3acd63ee0
S3_PINS_LADDER="$S3_RAMEN_N46_SHA  $S3_RAMEN_N46_PATH
$S3_RAMEN_N47_SHA  $S3_RAMEN_N47_PATH
$S3_RAMEN_N48_SHA  $S3_RAMEN_N48_PATH
$S3_FT_N46_SHA  $S3_FT_N46_PATH
$S3_FT_N47_SHA  $S3_FT_N47_PATH
$S3_FT_N48_SHA  $S3_FT_N48_PATH"                                              # `sha256sum -c` format, checked in every ladder job before the generator starts
S3_PINS_FILTER="$S3_FT_N46_SHA  $S3_FT_N46_PATH"
# ===== filter-bed declaration (FILTER_SETTINGS_DECLARATION.md): the FT spec and the exact driver arguments. `none` first (in-run reproduction row), `agf0.2t0.06` = the TAG step-0 row, then the 5 cited settings =====
S3_FT_SPEC="metric_vda_s_ft:$S3_FT_N46_PATH"
S3_GRID="none agf0.2t0.06 kal0.05 oneeuro1.0b1.0 ema0.7 tmed5 flowema0.7"
S3_F_EXPECT_FILES="bench=196,night=7,fnoise=7"                                  # the driver's own --plan output at 9f14d1e for S3_FT_SPEC and S3_GRID (dryruns/filters_plan.txt); b10-style 56/2/2 does not apply
S3_F_DERIVED=28                                                                 # kitti_minus_0047 x {llvar, llvarf, st_llvar, st_llb3} x 7 variants
# ===== ladder item / job constants =====
S3_ITEM_LADDER="tab:noise"                                                      # at I3: 12 models x kitti,bonn x 10 cells (llb3kappa1 / st_llb3kappa1 dropped), split dev, temporal true
S3_EXP_LADDER="ok=120,resumed=120,derived=0"                                    # 120 grid passes + 120 TAG group members = 240 units (plan: dryruns/plan_tab_noise.json)
S3_EXP_LADDER_SMOKE="ok=6,resumed=6,derived=0"                                  # smoke at I3: 2 base models x 3 cells (llb3x1, llb3x16, st_llb3x1) (+ 6 TAG members) = 12 files, kitti, 1 clip (dryruns/plan_tab_noise_smoke.json)
export S3_ROOT S3_CODE S3_DCODE S3_SCORER_CLONE S3_VDA_REPO S3_VENV S3_BVI
unset DATA_ROOT RAMEN_HOST BVI_ROOT RAMEN_SCORER_CLONE                   # the GH200 per-key map must be active (host detected from nvidia-smi); no local-style override
s3_check_placeholders() {   # returns 1 (message on stderr) while a placeholder is unfilled or malformed
  local bad=0
  [[ $S3_WANT =~ ^[0-9a-f]{40}$ ]] || { echo "PLACEHOLDER S3_WANT not filled (commit I3 sha): $S3_WANT" >&2; bad=1; }
  [[ $S3_CONFIG_SHA16 =~ ^[0-9a-f]{16}$ ]] || { echo "S3_CONFIG_SHA16 malformed: $S3_CONFIG_SHA16" >&2; bad=1; }
  [[ $S3_GEN_SHA16 =~ ^[0-9a-f]{16}$ ]] || { echo "S3_GEN_SHA16 malformed: $S3_GEN_SHA16" >&2; bad=1; }
  return $bad
}
