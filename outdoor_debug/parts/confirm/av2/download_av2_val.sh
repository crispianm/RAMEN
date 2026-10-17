#!/bin/sh
# AV2 Sensor VAL rule-selected subset -> Isambard login node. R1 (Auditor O43): phase-searching frame rule, meta in $META. Run `fetch` only after the Auditor OKs R1.
# Pulls ONLY: ring_front_center jpgs of the clip frames (every 2nd frame, <=160/log), the LiDAR sweep nearest to each scored frame, calibration/*, city_SE3_egovehicle.feather,
# for the 50 rule-selected logs (av2_val_logs_selected50.txt). Source: public bucket, no-sign-request, plain HTTPS (curl; no aws/s5cmd needed).
# Stages (run in order; each is idempotent/resumable):  list -> select -> fetch -> verify
#   AV2_REG_SHA=<sha256 of the FILED AV2_REGISTRATION.md>  is REQUIRED for list/select/fetch (registration-first guard; the sha is recorded in the manifest dir).
#   usage: AV2_REG_SHA=... sh download_av2_val.sh list|select|fetch|verify
# Login node: python3.6 default (select_av2_frames.py is stdlib-only and 3.6-safe); uv + python3.11 available for the XML listing parse (stdlib xml only too -> python3 is fine).
set -eu
META=${META:-${DEST:-/projects/b5dh/data/heldout/av2_val}/meta_R1}   # R1: new meta dir; the pre-R1 meta/ (392-frame selection) stays untouched
DEST=${DEST:-/projects/b5dh/data/heldout/av2_val}        # NEW directory; refuse if it exists with foreign content at 'list'
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=${LOGS:-$HERE/av2_val_logs_selected50.txt}
LOGS_SHA=f7f508f958e8e9d173772677d239c7a53165a0fb67520a68a4bfcf134592e370
BASE=https://argoverse.s3.amazonaws.com
PFX=datasets/av2/sensor/val
PAR=${PAR:-16}
stage=${1:?stage: list|select|fetch|verify}
[ "$stage" = verify ] || : "${AV2_REG_SHA:?set AV2_REG_SHA to the sha256 of the filed registration}"
[ "$(sha256sum "$LOGS" | cut -d' ' -f1)" = "$LOGS_SHA" ] || { echo "log list sha mismatch"; exit 2; }
mkdir -p "$META" "$DEST/listing" "$DEST/data"
echo "$AV2_REG_SHA" > "$META/registration_sha256.txt" 2>/dev/null || true

case "$stage" in
list)   # NAMES + SIZES only (S3 ListObjectsV2); no object body is requested
  while read -r log; do
    [ -s "$DEST/listing/$log.keys" ] && continue
    tok=""; : > "$DEST/listing/$log.tmp"
    while :; do
      url="$BASE/?list-type=2&prefix=$PFX/$log/&max-keys=1000"
      [ -n "$tok" ] && url="$url&continuation-token=$tok"
      curl -sS --retry 5 -m 120 "$url" -o "$DEST/listing/$log.xml"
      python3 - "$DEST/listing/$log.xml" >> "$DEST/listing/$log.tmp" <<'PY'
import sys, re
x = open(sys.argv[1]).read()
for m in re.finditer(r"<Contents><Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", x, re.S):
    print("%s\t%s" % (m.group(1), m.group(2)))
t = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
open(sys.argv[1] + ".tok", "w").write(t.group(1) if t else "")
PY
      tok=$(cat "$DEST/listing/$log.xml.tok"); [ -z "$tok" ] && break
      tok=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$tok")
    done
    mv "$DEST/listing/$log.tmp" "$DEST/listing/$log.keys"
  done < "$LOGS"
  echo "listed $(wc -l < "$LOGS") logs"; cat "$DEST"/listing/*.keys | wc -l ;;
select)
  python3 "$HERE/select_av2_frames.py" "$DEST/listing" "$LOGS" "$META/av2_frames.tsv" "$META/objects.tsv"
  sha256sum "$META/av2_frames.tsv" "$META/objects.tsv" | tee "$META/selection.sha256"
  echo "Review the printed total_bytes, then run fetch (needs CONFIRM_BYTES=<total_bytes>)" ;;
fetch)
  total=$(awk -F'\t' '{s+=$2} END{print s}' "$META/objects.tsv")
  [ "${CONFIRM_BYTES:-}" = "$total" ] || { echo "set CONFIRM_BYTES=$total after reviewing the size"; exit 3; }
  avail=$(df -B1 --output=avail "$DEST" | tail -1); [ "$avail" -gt $((total * 2)) ] || { echo "not enough disk"; exit 4; }
  cut -f1 "$META/objects.tsv" | xargs -P "$PAR" -I{} sh -c '
    k="{}"; o="'"$DEST"'/data/${k#datasets/av2/sensor/val/}"; mkdir -p "$(dirname "$o")"
    [ -s "$o" ] && exit 0
    curl -sS --retry 8 --retry-delay 3 -m 600 -f "'"$BASE"'/$k" -o "$o.part" && mv "$o.part" "$o" || echo "FAIL $k" >> "'"$META"'/fetch_fail.txt"' 
  [ ! -s "$META/fetch_fail.txt" ] || { echo "FAILURES:"; wc -l "$META/fetch_fail.txt"; exit 5; }
  echo fetch done ;;
verify)
  # 1) every object present with the LISTED size; 2) sha256 manifest; 3) counts
  bad=0
  while IFS="$(printf '\t')" read -r k s; do
    o="$DEST/data/${k#datasets/av2/sensor/val/}"
    [ -f "$o" ] && [ "$(stat -c %s "$o")" = "$s" ] || { echo "BAD $k"; bad=$((bad+1)); }
  done < "$META/objects.tsv"
  [ "$bad" = 0 ] || { echo "$bad objects bad"; exit 6; }
  # manifest = the R1 object list ONLY (pre-R1 unused files and any leftover .part are excluded by construction)
  ( cd "$DEST/data" && cut -f1 "$META/objects.tsv" | sed "s#^datasets/av2/sensor/val/#./#" | sort | xargs -P 8 -n 50 sha256sum | sort -k2 > "$META/SHA256SUMS" )
  n_obj=$(wc -l < "$META/objects.tsv"); n_sum=$(wc -l < "$META/SHA256SUMS")
  [ "$n_obj" = "$n_sum" ] || { echo "object count $n_obj != manifest $n_sum"; exit 7; }
  sha256sum "$META/SHA256SUMS" | tee "$META/SHA256SUMS.sha256"
  echo "frames: $(wc -l < "$META/av2_frames.tsv")  logs: $(ls "$DEST/data" | wc -l)  jpg: $(find "$DEST/data" -name '*.jpg' | wc -l)  lidar: $(find "$DEST/data/" -path '*sensors/lidar*' -name '*.feather' | wc -l)" ;;
*) echo "unknown stage"; exit 1 ;;
esac
