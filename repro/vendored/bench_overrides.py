"""Run bench_joint.py (any scorer checkout) with the C3 evaluation overrides coupling_scale / stage2_scale set on the loaded net.
No scorer or checkpoint file is changed; it wraps bench_joint.load_ours exactly like scripts/dn01_headc_engage.py.
Usage (cwd = scorer root, so `import bench_joint` resolves there):
  python <this file> --coupling 0 --stage2 1 -- run --model ours:<ckpt> --dataset kitti --cells llvar st_llvar --split dev --temporal --grid none --diag --device cuda --out <DIR>
Everything after `--` is bench_joint's own CLI. Use a DIFFERENT --out per setting (result file names do not contain the override).
An OVERRIDES.json sidecar is written into --out so a result dir always says which setting produced it."""
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--coupling", type=float, default=1.0)
ap.add_argument("--stage2", type=float, default=1.0)
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
sys.path.insert(0, str(Path.cwd()))
import bench_joint as bj  # noqa: E402

_orig = bj.load_ours


def _patched(ckpt, device):
    net = _orig(ckpt, device)
    if getattr(net, "depth_input", "raw") != "restored":
        raise SystemExit("overrides need a depth_input='restored' (coupled) checkpoint")
    net.coupling_scale, net.stage2_scale = a.coupling, a.stage2
    print(f"[overrides] coupling_scale={a.coupling} stage2_scale={a.stage2} g={float(net.gate.g):.4f} "
          f"g2={float(net.gate2.g) if net.gate2 is not None else None}", flush=True)
    return net


bj.load_ours = _patched
if "--out" in rest:
    od = Path(rest[rest.index("--out") + 1]); od.mkdir(parents=True, exist_ok=True)
    (od / "OVERRIDES.json").write_text(json.dumps({"coupling_scale": a.coupling, "stage2_scale": a.stage2, "argv": rest}))
sys.argv = ["bench_joint.py"] + rest
bj.main()
