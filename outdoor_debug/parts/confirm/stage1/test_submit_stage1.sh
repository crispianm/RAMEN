#!/bin/bash
# Static checks of submit_stage1.sh (T5 blocker, Auditor O53 / Manager): NO sbatch --dependency (afterok/any, in the =, space and -d forms) from a smoke (SM1/SM2) to any job, an explicit by-hand GATE before the fulls,
# and FORCED-FAIL controls: each mutated copy of the script must be detected. Work dir: the job tmp (never shared /tmp).
set -uo pipefail
H="$(cd "$(dirname "$0")" && pwd)"; S="$H/submit_stage1.sh"; W="${ST1_SUBMIT_TEST_DIR:-/home/wg19671/.claude/jobs/a0d391cb/tmp/st1_submit_test}"; rm -rf "$W"; mkdir -p "$W"; ok=1
RX='(dependency[= ]|-d )[a-z]+:[^ ]*\$\{?SM[0-9]'
chk() { if [[ "$2" == "$3" ]]; then echo "OK    $1"; else echo "WRONG $1 (got $2, want $3)"; ok=0; fi; }
chk "submit_stage1.sh: no sbatch --dependency from any smoke (SM1/SM2)" "$(grep -E "$RX" "$S" | wc -l)" 0
chk "submit_stage1.sh: the six full sbatch lines carry no --dependency at all" "$(grep -E '^\s*sbatch --parsable .*TRAIN_S1_n\$\{N\}_\$\{A\}' "$S" | grep -c dependency)" 0
chk "submit_stage1.sh: the explicit GATE step is present before Block 2" "$(awk '/>>> GATE/{g=NR} /^### Block 2/{b=NR} END{print (g>0 && b>g) ? 1 : 0}' "$S")" 1
mut() { sed -e "$1" "$S" > "$W/m.sh"; }
FULL='s#^  sbatch --parsable --export=ALL,PROJECT_DIR=\$CL#  sbatch --parsable DEPFLAG --export=ALL,PROJECT_DIR=$CL#'
for f in '--dependency=afterok:$SM1:$SM2' '--dependency afterok:$SM1:$SM2' '-d afterok:$SM1' '--dependency=afterany:$SM2'; do
  mut "${FULL/DEPFLAG/$f}"; chk "CONTROL mutated full-run line with '$f' is detected" "$(grep -E "$RX" "$W/m.sh" | wc -l)" 1
done
mut 's|^### >>> GATE|### no gate|'; chk "CONTROL removing the GATE marker is detected" "$(awk '/>>> GATE/{g=NR} /^### Block 2/{b=NR} END{print (g>0 && b>g) ? 1 : 0}' "$W/m.sh")" 0
bash -n "$S" && echo "OK    bash -n" || { echo "WRONG bash -n"; ok=0; }
[[ $ok -eq 1 ]] && echo "STAGE1 SUBMIT SUITE: ALL AS EXPECTED" || echo "STAGE1 SUBMIT SUITE: UNEXPECTED RESULT"; exit $((1-ok))
