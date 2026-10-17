import sys, json, numpy as np
d, out, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od_a4/outdoor_debug/parts/scorer"); sys.path.insert(0, d); sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od_a4")
import torch, bench_joint as bj, ddad_scorer as DS
assert DS.__file__.startswith(d), DS.__file__
bj.build_pipeline = lambda spec, dev: {"metric": True}
bj.run_pipeline = lambda p, x, s, b, hw: (torch.full((x.shape[0], *x.shape[-2:]), 7.0, device=x.device), None)
DS.model_meta = lambda c, s, p: (s, dict(spec=s)); DS.assert_pins = lambda *a, **k: {}
DS.D.read_frame_list = lambda *a, **k: None; DS.D.read_pairs = lambda *a, **k: None
DS.D.load_clips = lambda lim=None: [("s0", [dict(image="a.png", scored=False)])]
DS.D.load_rgb = lambda p: np.full((1216, 1936, 3), 128, np.uint8)
DS.main(["--spec", "metric_vda_s", "--out", out, "--device", "cuda", "--limit-scenes", "1", *extra])
r = json.load(open(out))
for k in ("seconds", "ddad_scorer_sha256", "host", "ramen_commit"): r["meta"].pop(k, None)
json.dump(r, open(out, "w"), indent=1, sort_keys=True)
