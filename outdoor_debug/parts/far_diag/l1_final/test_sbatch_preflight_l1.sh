#!/bin/bash
# Dry-run of the TRAIN sbatch PREFLIGHT (not the training) on a local clone, with forced-fail controls. The sbatch is copied with ONLY environment constants swapped (cluster paths -> local paths,
# torchrun -> echo, the post-training checksum block removed); every check under test is the committed logic. usage: test_sbatch_preflight_l1.sh <pin-commit-sha>
# needs ~/.venvs/ramen and the local release weights; works in $L1_PF_DIR (default: the job tmp dir; never shared /tmp). Exit 0 iff every control behaved as expected.
set -uo pipefail
PIN="$1"; SRC="$(cd "$(dirname "$0")/../../../.." && pwd)"; W="${L1_PF_DIR:-/home/wg19671/.claude/jobs/a0d391cb/tmp/l1_pf}"; rm -rf "$W"; mkdir -p "$W/sb" "$W/cache"
git clone -q "$SRC" "$W/clone" && git -C "$W/clone" checkout -q --detach "$PIN" && mkdir -p "$W/clone/logs"
D=outdoor_debug/parts/far_diag/l1_final; ok=1; REL_LOCAL=/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth
git -C "$W/clone" config user.email t@t; git -C "$W/clone" config user.name t
mk() {  # mk <ARM> <K> <pin> [extra sed -e args...]  -> $W/sb/cur.sbatch
  local a="$1" k="$2" pin="$3"; shift 3
  sed -e "s/UNSET_PIN_COMMIT/$pin/" -e "s#/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth#$REL_LOCAL#" -e "s#/projects/b5dh/.torch_cache#$W/cache#" -e "s#/projects/b5dh/FocusMamba/.venv#$HOME/.venvs/ramen#" \
      -e 's#^torchrun .*#echo DRYRUN-torchrun#' -e 's#^source "\$VENV/bin/activate"#export PATH="$HOME/.venvs/ramen/bin:$PATH"#' -e '/^FINAL=/,/^\[\[ -f "\$FINAL"/d' "$@" "$SRC/$D/sbatch/TRAIN_${a}_k${k}.sbatch" > "$W/sb/cur.sbatch"; }
go() {  # go <label> <want 0|1> <substring or ''> [ENV=..]...   (runs $W/sb/cur.sbatch in the clone)
  local label="$1" want="$2" sub="$3"; shift 3
  ( cd "$W/clone" && env PROJECT_DIR="$W/clone" "$@" bash "$W/sb/cur.sbatch" ) > "$W/out.txt" 2>&1; local rc=$?; [[ $rc -ne 0 ]] && rc=1
  local g=1; [[ -n "$sub" ]] && ! grep -qF -- "$sub" "$W/out.txt" && g=0
  if [[ $rc -eq $want && $g -eq 1 ]]; then echo "OK    $label (rc=$rc${sub:+, log has '$sub'})"; else echo "WRONG $label (rc=$rc want $want; substring '$sub' found=$g)"; tail -3 "$W/out.txt" | cut -c1-220; ok=0; fi
}
clean() { rm -rf "$W/clone/runs"; git -C "$W/clone" checkout -q --detach "$PIN"; git -C "$W/clone" reset -q --hard "$PIN"; }
BAD=0123456789abcdef0123456789abcdef01234567
clean; mk B0 2000 "$PIN";                          go "POSITIVE B0 k2000 full: preflight passes and reaches the launch" 0 "DRYRUN-torchrun"
clean; mk S5 4000 "$PIN";                          go "POSITIVE S5 k4000 SMOKE=120: smoke config + dir" 0 "runs/l1_smoke_S5_k4000" SMOKE=120
echo "      smoke diff lines in log: $(grep -c '^>' "$W/out.txt") (must be 3)  $(grep '^>' "$W/out.txt" | tr '\n' ';')"; [[ "$(grep -c '^>' "$W/out.txt")" == 3 ]] || ok=0
clean; mk B0 2000 UNSET_PIN_COMMIT;                go "NEG PIN placeholder not substituted" 1 "not a 40-hex"
clean; mk B0 2000 "$BAD";                          go "NEG PIN is another sha (HEAD != PIN)" 1 "!= PIN_COMMIT"
clean; mk B0 2000 "$PIN"; echo x >> "$W/clone/README.md"; go "NEG dirty working tree" 1 "working tree dirty"; git -C "$W/clone" checkout -q -- README.md
clean; mk B0 2000 "$PIN" -e 's/^export CFG_SHA=.*/export CFG_SHA="0000000000000000000000000000000000000000000000000000000000000000"/'; go "NEG config sha mismatch" 1 "config sha"
clean; mk B0 2000 "$PIN" -e 's/^export REL_SHA=.*/export REL_SHA="0000000000000000000000000000000000000000000000000000000000000000"/'; go "NEG release weights sha mismatch" 1 "release sha"
clean; mk B0 2000 "$PIN" -e 's/^export SEED=.*/export SEED="49"/'; go "NEG seed in sbatch (49) != data.seed in the B0 config (46)" 1 "data.seed 49 not in config"
clean; mkdir -p "$W/clone/runs/l1_S3_k2000"; mk S3 2000 "$PIN"; go "NEG output dir already exists" 1 "already exists"
# S6 gate
PSHA=$(sha256sum "$SRC/$D/S6_diff.patch" | cut -d' ' -f1)
clean; mk S6 2000 "$PIN";                          go "NEG S6 without S6_CLEARED_PATCH_SHA" 1 "needs S6_CLEARED_PATCH_SHA"
clean; mk S6 2000 "$PIN";                          go "NEG S6 with a wrong cleared sha" 1 " != the cleared sha 0000000000000000000000000000000000000000000000000000000000000000" S6_CLEARED_PATCH_SHA=0000000000000000000000000000000000000000000000000000000000000000
clean; mk S6 2000 "$PIN";                          go "POSITIVE S6 with the cleared patch sha (train.py == base + patch)" 0 "S6 gate OK" S6_CLEARED_PATCH_SHA=$PSHA
clean; mk S6 2000 "$PIN";                          go "POSITIVE S6 gate under git core.abbrev=7 (host-independent diff text)" 0 "S6 gate OK" S6_CLEARED_PATCH_SHA=$PSHA GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.abbrev GIT_CONFIG_VALUE_0=7
clean; mk S6 2000 "$PIN";                          go "POSITIVE S6 gate under git core.abbrev=12" 0 "S6 gate OK" S6_CLEARED_PATCH_SHA=$PSHA GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.abbrev GIT_CONFIG_VALUE_0=12
clean; mk S6 2000 "$PIN" -e 's/^export TRAIN_PY_SHA=.*/export TRAIN_PY_SHA="0000000000000000000000000000000000000000000000000000000000000000"/'; go "NEG S6: wrong TRAIN_PY_SHA literal" 1 "train.py sha != TRAIN_PY_SHA" S6_CLEARED_PATCH_SHA=$PSHA
clean; echo "# stale edit" >> "$W/clone/train.py"; git -C "$W/clone" commit -q -am "stale train.py"; NP=$(git -C "$W/clone" rev-parse HEAD); mk S6 2000 "$NP"; go "NEG S6: committed train.py differs from base+cleared patch (stale/edited train.py)" 1 "differs from base+cleared patch" S6_CLEARED_PATCH_SHA=$PSHA
clean; mk S6 2000 "$PIN" -e 's/^export TRAIN_BASE_COMMIT=.*/export TRAIN_BASE_COMMIT="HEAD~1"/'; go "NEG S6: train.py differs from base+cleared patch (base moved)" 1 "differs from base+cleared patch" S6_CLEARED_PATCH_SHA=$PSHA
# B2 (second review + Auditor O51): patch AND train.py edited CONSISTENTLY in the clone (patch regenerated from the edited train.py, TRAIN_PY_SHA set to the edited train.py's sha in the sbatch copy).
# Only the Auditor-cleared LITERAL can catch this: the clone's own patch sha would pass (the hole), the literal FAILS and the message names it.
LIT=298cac3bde8289dafc6b9a37c308e7a69819d9a95e49733702cbd7231d55ff26; BASE=8e1d9ae
clean; echo "# consistent edit" >> "$W/clone/train.py"; git -C "$W/clone" commit -q -am "edit train.py"
git -C "$W/clone" diff --full-index --no-color --no-ext-diff $BASE HEAD -- train.py > "$W/clone/$D/S6_diff.patch"; git -C "$W/clone" commit -q -am "regenerate patch consistently"
NP2=$(git -C "$W/clone" rev-parse HEAD); NT=$(sha256sum "$W/clone/train.py" | cut -d' ' -f1); NPS=$(sha256sum "$W/clone/$D/S6_diff.patch" | cut -d' ' -f1)
mk S6 2000 "$NP2" -e "s/^export TRAIN_PY_SHA=.*/export TRAIN_PY_SHA=\"$NT\"/"
go "HOLE DEMO: consistent edit passes with the clone-derived patch sha (what the old submit_l1.sh S6SHA did)" 0 "S6 gate OK" S6_CLEARED_PATCH_SHA=$NPS
go "NEG B2: consistent train.py+patch edit FAILS on the Auditor literal 298cac3b, the message names it" 1 "!= the cleared sha $LIT" S6_CLEARED_PATCH_SHA=$LIT
# mutated-config commits (the yaml is changed, committed, its sha put in the sbatch: only the semantic greps can catch these)
mutc() {  # mutc <ARM> <K> <label> <want substring> <python edit on yaml text>
  clean; local y="$W/clone/$D/configs/k$2/arm_$1.yaml"; python3 -c "
import sys; p=sys.argv[1]; t=open(p).read(); n=t$5; assert n!=t, 'edit did not apply'; open(p,'w').write(n)" "$y" || { echo "WRONG $3 (edit failed)"; ok=0; return; }
  git -C "$W/clone" commit -q -am mut; local np; np=$(git -C "$W/clone" rev-parse HEAD); local ys; ys=$(sha256sum "$y" | cut -d' ' -f1)
  mk "$1" "$2" "$np" -e "s/^export CFG_SHA=.*/export CFG_SHA=\"$ys\"/"; go "$3" 1 "$4"; }
mutc B0 2000 "NEG yaml with resume_from set" "resume_from is not null" ".replace('  resume_from: null','  resume_from: /x.pt')"
mutc B0 2000 "NEG yaml with a resume_reseed key" "resume_reseed" ".replace('  resume_from: null','  resume_from: null\\n  resume_reseed: 5')"
mutc S2 2000 "NEG yaml with lr_restart keys" "resume_reseed / lr_restart" ".replace('  resume_from: null','  resume_from: null\\n  lr_restart_step: 12000')"
mutc S4 4000 "NEG yaml with max_steps 2000 under a k4000 name" "lacks 'max_steps: 4000'" ".replace('  max_steps: 4000','  max_steps: 2000')"
mutc S1 2000 "NEG yaml with vda_init not released" "vda_init is not 'released'" ".replace('  vda_init: released','  vda_init: scratch')"
mutc B0 2000 "NEG yaml with warmup 100" "warmup_steps: 500" ".replace('  warmup_steps: 500','  warmup_steps: 100')"
[[ $ok -eq 1 ]] && echo "PREFLIGHT DRY-RUN SUITE: ALL AS EXPECTED" || echo "PREFLIGHT DRY-RUN SUITE: UNEXPECTED RESULT"; exit $((1-ok))
