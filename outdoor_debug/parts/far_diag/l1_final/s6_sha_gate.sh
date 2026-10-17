#!/bin/bash
# usage: s6_sha_gate.sh <cleared-sha256 (64 lowercase hex, pasted from the Auditor's record)> <path/to/S6_diff.patch>
# Exit 0 iff the value is a real 64-hex sha256 (no placeholder) AND equals sha256(patch). The cleared value must come from OUTSIDE the clone (submit_l1.sh holds the literal).
set -uo pipefail
C="${1:-}"; P="${2:-}"
[[ "$C" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: cleared S6 sha '$C' is not 64 lowercase hex (placeholder?)"; exit 1; }
[[ "$C" =~ ^(.)\1{63}$ ]] && { echo "ERROR: cleared S6 sha '$C' is a degenerate placeholder"; exit 1; }
[[ -f "$P" ]] || { echo "ERROR: patch '$P' not found"; exit 1; }
G="$(sha256sum "$P" | cut -d' ' -f1)"
[[ "$G" == "$C" ]] || { echo "ERROR: sha256($P) = $G != the Auditor-cleared literal $C"; exit 1; }
echo "S6 patch sha OK: $G == the cleared literal"
