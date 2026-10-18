#!/bin/bash
# Structure + forced-fail controls for submit_l1.sh and s6_sha_gate.sh (second review B2, S1; Auditor O51). Scratch in $L1_SUB_DIR (default: the job tmp dir). Exit 0 iff every control behaved.
set -uo pipefail
H="$(cd "$(dirname "$0")" && pwd)"; W="${L1_SUB_DIR:-/home/wg19671/.claude/jobs/a0d391cb/tmp/l1_sub}"; rm -rf "$W"; mkdir -p "$W"; ok=1; n=0
LIT=298cac3bde8289dafc6b9a37c308e7a69819d9a95e49733702cbd7231d55ff26
chk() { n=$((n+1)); if [[ "$2" == "$3" ]]; then echo "OK    $1"; else echo "WRONG $1 (got $2 want $3)"; ok=0; fi; }
gate() { bash "$H/s6_sha_gate.sh" "$@" > "$W/g.out" 2>&1; echo $?; }
# ---- s6_sha_gate.sh ----
chk "gate POSITIVE: literal == sha256(S6_diff.patch)" "$(gate $LIT $H/S6_diff.patch)" 0
chk "gate NEG: placeholder '<64-hex>'" "$(gate '<64-hex>' $H/S6_diff.patch)" 1
chk "gate NEG: empty value" "$(gate '' $H/S6_diff.patch)" 1
chk "gate NEG: 64 zeros (degenerate placeholder)" "$(gate $(printf '0%.0s' $(seq 64)) $H/S6_diff.patch)" 1
chk "gate NEG: uppercase / 63 hex" "$(gate ${LIT:1} $H/S6_diff.patch)" 1
cp "$H/S6_diff.patch" "$W/edited.patch"; echo "# edited" >> "$W/edited.patch"
chk "gate NEG (the second review's B2): a patch edited after clearing (literal != sha256)" "$(gate $LIT $W/edited.patch)" 1
grep -q "$LIT" "$W/g.out" && c=1 || c=0; chk "gate failure message names the literal" "$c" 1
chk "gate NEG: missing patch file" "$(gate $LIT $W/none.patch)" 1
chk "gate: the self-derived sha of the EDITED patch would pass the gate (this is the hole the literal closes)" "$(gate $(sha256sum $W/edited.patch | cut -d' ' -f1) $W/edited.patch)" 0
# ---- submit_l1.sh structure ----
S="$H/submit_l1.sh"
chk "submit_l1.sh: S6SHA is the Auditor's literal" "$(grep -c "^export S6SHA=$LIT\$" $S)" 1
chk "submit_l1.sh: no S6SHA derived by sha256sum" "$(grep -E 'S6SHA=.*(sha256sum|\$\()' $S | wc -l)" 0
chk "submit_l1.sh: the literal gate is called" "$(grep -c 's6_sha_gate.sh "\$S6SHA"' $S)" 1
chk "submit_l1.sh: no sbatch --dependency from any smoke (SM1/SM5/SM6) to a job" "$(grep -E '(dependency[= ]|-d )[a-z]+:[^ ]*\$\{?SM[0-9]' $S | wc -l)" 0
chk "submit_l1.sh: J_S6 has no dependency" "$(grep -E '^J_S6=.*dependency' $S | wc -l)" 0
chk "submit_l1.sh: GATE 1 and GATE 2 are explicit steps" "$(grep -c '>>> GATE [12]' $S)" 2
chk "submit_l1.sh: all smoke probes use --smoke 120" "$(grep -o 'step120.pt --arm S[0-9] -K 2000 --smoke 120' $S | wc -l)" 3
chk "submit_l1.sh: bash -n" "$(bash -n $S 2>&1 | wc -l)" 0
# forced-fail: the structure checks must fail on bad variants
mut() { sed -e "$1" "$S" > "$W/m.sh"; }
mut 's#^export S6SHA=.*#export S6SHA=$(sha256sum $D/S6_diff.patch | cut -d" " -f1)#'; chk "CONTROL old derived S6SHA is detected" "$(grep -c "^export S6SHA=$LIT\$" $W/m.sh)$(grep -E 'S6SHA=.*(sha256sum|\$\()' $W/m.sh | wc -l)" 01
mut 's#^B0=$(sbatch --parsable #B0=$(sbatch --parsable --dependency=afterok:$SM1:$SM5 #'; chk "CONTROL B0 afterok smokes is detected" "$(grep -E '(dependency[= ]|-d )[a-z]+:[^ ]*\$\{?SM[0-9]' $W/m.sh | wc -l)" 1
mut 's#^J_S6=$(sbatch --parsable #J_S6=$(sbatch --parsable --dependency=afterok:$SM6 #'; chk "CONTROL J_S6 afterok SM6 is detected" "$(grep -E '(dependency[= ]|-d )[a-z]+:[^ ]*\$\{?SM[0-9]' $W/m.sh | wc -l)" 1
mut 's#^J_S6=$(sbatch --parsable #J_S6=$(sbatch --parsable --dependency afterok:$SM6 #'; chk "CONTROL J_S6 '--dependency afterok:' (space form) is detected" "$(grep -E '(dependency[= ]|-d )[a-z]+:[^ ]*\$\{?SM[0-9]' $W/m.sh | wc -l)" 1
mut 's#^J_S6=$(sbatch --parsable #J_S6=$(sbatch --parsable -d afterok:$SM6 #'; chk "CONTROL J_S6 '-d afterok:' form is detected" "$(grep -E '(dependency[= ]|-d )[a-z]+:[^ ]*\$\{?SM[0-9]' $W/m.sh | wc -l)" 1
mut 's#--smoke 120 --expect-step 120 -F"#--expect-step 120 -F"#'; chk "CONTROL a probe without --smoke is detected" "$(grep -o 'step120.pt --arm S[0-9] -K 2000 --smoke 120' $W/m.sh | wc -l)" 2
[[ $ok -eq 1 ]] && echo "SUBMIT SUITE: ALL AS EXPECTED ($n checks)" || echo "SUBMIT SUITE: UNEXPECTED RESULT"; exit $((1-ok))
