"""DN-01 pre-launch gate (a) for the re-calibrated distilled temp2 (Manager / Auditor 2026-09-26, after the BN-recal
defect, session_briefs/distill_defect_20260926.md). Thresholds are fixed here, committed BEFORE the gate runs:
  GATE (a): on N fresh draws of the TRAINING mix (the config's train loader, seed --seed != the distillation's 42;
            crops --crop^2 after full-res degradation, as distilled), the student's EVAL-mode L1 to the teacher (stored
            BN statistics, as the ST models run) <= EVAL_OVER_TRAIN_MAX x its TRAIN-mode L1 (per-clip batch
            statistics, as it was distilled). A stored-statistics defect shows up as eval >> train (0.0523 vs 0.0156
            for the broken full; 0.0147 vs 0.0156 after the hunt's local fix).
Gate (b) (restoration floors vs Pocket on bench_joint dev clean + llb3) is checked on the step-0 JSONs by
provenance/dn01_st_step0/compare_st.py.
  python scripts/dn01_st_recal_gate.py --config configs/experiments/dn01_ll_ctrl.yaml --ckpt runs/recal_full/temp2_distilled.pt \
      --out runs/recal_full/gate_a.json      (exit 1 on FAIL)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_OVER_TRAIN_MAX = 1.10
N_CLIPS = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=N_CLIPS)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--crop", type=int, default=252)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import bench_joint as bj
    from models.restoration import PocketFrontEnd
    from models.shared_temporal import temp2_block
    from scripts.dn01_st_distill import _y
    from train import _set_dataset_attr, build_dataloader, crop_batch, load_config, seed_everything
    cfg = load_config(Path(a.config))
    cfg["data"]["seed"] = a.seed
    seed_everything(a.seed)
    loader, _ = build_dataloader(cfg, "train")
    d = cfg.get("data", {})
    for k in ("clean_prob", "seeded_noise", "blur_only_prob", "blur_disk_prob", "lowlight"):
        _set_dataset_attr(loader.dataset, k, d.get(k, 0.0 if k.endswith("prob") else None))
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    teacher = PocketFrontEnd().to(a.device).eval()
    ev = temp2_block(ck["width"]).to(a.device)
    ev.load_state_dict(ck["temp2"], strict=True)
    ev.eval()
    tr = copy.deepcopy(ev).train()
    for m in tr.modules():                                   # train-mode BN (batch stats), running stats untouched
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 0.0
    gen = torch.Generator().manual_seed(a.seed)
    l_ev, l_tr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for b in loader:
            for c in crop_batch(b, a.crop, gen)["degraded_frames"].to(a.device).permute(0, 2, 1, 3, 4):
                t = teacher(c)
                l_ev += float((_y(ev, c) - t).abs().mean())
                l_tr += float((_y(tr, c) - t).abs().mean())
                n += 1
            if n >= a.clips:
                break
    l_ev, l_tr = l_ev / n, l_tr / n
    ok = l_ev <= EVAL_OVER_TRAIN_MAX * l_tr
    art = {"gate": "a: eval-mode L1 <= %.2f x train-mode L1 to the teacher" % EVAL_OVER_TRAIN_MAX, "pass": ok,
           "l1_eval": l_ev, "l1_train": l_tr, "ratio": l_ev / l_tr, "clips": n, "seed": a.seed, "crop": a.crop,
           "ckpt": {"path": a.ckpt, "sha256": bj.sha256_file(a.ckpt), "width": ck["width"],
                    "recal_mode": ck.get("recal_mode")}, "config": a.config, "ramen_commit": bj.git_state(bj.REPO)}
    Path(a.out).write_text(json.dumps(art, indent=1))
    print(f"GATE(a) {'PASS' if ok else 'FAIL'} {ck['width']}: eval L1 {l_ev:.5f} train L1 {l_tr:.5f} "
          f"ratio {l_ev / l_tr:.3f} (max {EVAL_OVER_TRAIN_MAX}) over {n} clips")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
