"""Diff two synthetic e2e dumps (dump_synth_outputs.py): every field outside meta must be identical; meta keys that differ are listed (allowed: provenance / new keys)."""
import json, sys
a, b = (json.load(open(p))["runs"] for p in sys.argv[1:3])
bad = 0
for arm in a:
    ra, rb = a[arm], b[arm]
    for k in sorted(set(ra) | set(rb)):
        if k == "meta":
            continue
        same = ra.get(k) == rb.get(k)
        print(f"{arm}.{k}: {'IDENTICAL' if same else 'DIFFERENT'}"); bad += not same
    ma, mb = ra["meta"], rb["meta"]
    ALLOWED = {"pins", "scorer_sha", "ckpt_sha", "registry_sha", "cfg", "cfg_sha256", "seconds", "host", "tag"}
    ck = lambda m: {k: v for k, v in m["ckpt"].items() if k not in ("path", "sha256")}      # temp path / file sha differ per run dir
    if ck(ma) != ck(mb):
        print("  meta.ckpt: UNEXPECTED DIFFERENCE"); bad += 1
    for k in sorted(set(ma) | set(mb)):
        if k == "ckpt":
            continue
        if ma.get(k) != mb.get(k):
            ok = k in ALLOWED or k not in ma
            print(f"  meta.{k}: {'changed/new (allowed provenance)' if ok else 'UNEXPECTED DIFFERENCE'}"); bad += not ok
print("RESULT:", "refusal-only: all numeric outputs identical" if not bad else f"{bad} unexpected differences")
sys.exit(bool(bad))
