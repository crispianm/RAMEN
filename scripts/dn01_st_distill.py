"""DN-01 final-v2 distillation stage for the shared-temporal arms (Manager D2; Auditor K39(b)). Restoration-path
COMPRESSION, NOT the clean-teacher feature distillation the user rejected on 2026-09-24: a temp2-architecture student
(models.shared_temporal.temp2_block, full or slim width, RANDOM init for both widths -- the same procedure, so width is
the only difference) learns to reproduce the full released PocketDVDNet v1.0 (sha256 78320cda..., PocketFrontEnd, BN in
eval mode) on the 5-frame window, from the raw (t-1, t, t+1) triplet it will see in the ST model. L1 on the student's
y = frame_t - residual vs the teacher's output. Data = the training mix of the config (C-LL's: awgn_blur + calibrated
lowlight), each step's clips cropped to --crop^2 after the full-resolution degradation (train.crop_batch). AdamW
(--lr, wd 0), --steps. Then the student's BatchNorm statistics are RECALIBRATED: reset, cumulative average over
--recal-batches clips of the same TRAINING mix (never dev / held-out clips; no gradients), and frozen from then on (the
ST models keep BN in eval mode).
FIX 2026-09-26 (independent hunt, session_briefs/distill_defect_20260926.md): the original recal ran ONE clip per
forward, so the stored statistics were the average of PER-CLIP statistics and missed the large between-clip variance of
this mix (clean / AWGN+blur / EV -3..-0.7); FULL's deep layers then sat 12-14 sigma off in eval mode (~18 dB on clean).
--recal-mode mixed (default) pools the triplets of --recal-mix clips, shuffles them and forwards 32 at a time, so the
statistics are POPULATION statistics; per_clip keeps the old recipe for comparison. --recal-from <temp2_distilled.pt>
re-calibrates existing weights without training (conv / affine parameters unchanged, only BN buffers).
  python scripts/dn01_st_distill.py --config configs/experiments/dn01_ll_ctrl.yaml --width slim --out runs/distill_slim
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _residual(blk: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x0 = blk.input_block(x)
    x1 = blk.down_0(x0)
    x2 = blk.up_1(blk.down_1(x1))
    x1 = blk.up_0(x1 + x2)
    return blk.output_block(x0 + x1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--width", choices=["full", "slim"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=252)
    ap.add_argument("--recal-batches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recal-mode", choices=["mixed", "per_clip"], default="mixed")
    ap.add_argument("--recal-mix", type=int, default=8)
    ap.add_argument("--recal-from", default=None, help="re-calibrate this temp2_distilled.pt's weights; no training")
    a = ap.parse_args()
    from models.restoration import PocketFrontEnd
    from models.shared_temporal import temp2_block
    from train import _set_dataset_attr, build_dataloader, crop_batch, load_config, seed_everything
    import bench_joint as bj
    cfg = load_config(Path(a.config))
    seed_everything(a.seed)
    loader, _ = build_dataloader(cfg, "train")
    d = cfg.get("data", {})
    for k in ("clean_prob", "seeded_noise", "blur_only_prob", "blur_disk_prob", "lowlight"):
        _set_dataset_attr(loader.dataset, k, d.get(k, 0.0 if k.endswith("prob") else None))
    teacher = PocketFrontEnd().to(a.device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    torch.manual_seed(a.seed)
    student = temp2_block(a.width).to(a.device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log, step, t0 = [], 0, time.time()

    def batches():
        while True:
            for b in loader:
                yield crop_batch(b, a.crop, gen)
    it = batches()
    src = None
    if a.recal_from:
        src = torch.load(a.recal_from, map_location="cpu", weights_only=False)
        assert src["width"] == a.width, (src["width"], a.width)
        student.load_state_dict(src["temp2"], strict=True)
        a.steps = 0
    while step < a.steps:
        b = next(it)
        x = b["degraded_frames"].to(a.device).permute(0, 2, 1, 3, 4)          # (B,T,3,h,w)
        with torch.no_grad():
            tgt = torch.stack([teacher(c) for c in x])
        pred = torch.stack([_y(student, c) for c in x])                        # student on raw triplets
        loss = (pred - tgt).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            log.append({"step": step, "l1": float(loss.detach()), "s": time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
    clips = (c for b in iter(lambda: next(it), None) for c in b["degraded_frames"].permute(0, 2, 1, 3, 4))
    recalibrate_bn(student, clips, a.recal_batches, a.recal_mode, a.recal_mix, gen, a.device)
    ck = out / "temp2_distilled.pt"
    torch.save({"temp2": student.state_dict(), "width": a.width, "lr": a.lr, "crop": a.crop,
                "steps": src["steps"] if src else a.steps, "recal_mode": a.recal_mode, "recal_mix": a.recal_mix,
                "recal_from": ({"path": a.recal_from, "sha256": bj.sha256_file(a.recal_from)} if src else None),
                "recal_batches": a.recal_batches, "seed": a.seed, "config": a.config,
                "teacher": {"pocketdvdnet_v1.0": teacher.checkpoint, "sha256": bj.sha256_file(teacher.checkpoint)},
                "ramen_commit": bj.git_state(bj.REPO)}, ck)
    (out / "distill_log.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {ck}  final l1 {log[-1]['l1'] if log else float('nan'):.5f}")


@torch.no_grad()
def recalibrate_bn(student: nn.Module, clips, n_clips: int, mode: str = "mixed", mix: int = 8,
                   gen: torch.Generator | None = None, device: str = "cpu", batch: int = 32) -> None:
    """Reset the student's BatchNorm running statistics and re-estimate them as a cumulative average (momentum None)
    over ``n_clips`` clips (T,3,h,w) from ``clips``; then momentum back to 0.1. ``mixed``: each forward is ``batch``
    triplets drawn from a shuffled pool of ``mix`` clips (population statistics); ``per_clip``: one clip per forward
    (the original recipe, which misses the between-clip variance)."""
    from models.shared_temporal import triplets
    for m in student.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    student.train()
    done = 0
    while done < n_clips:
        k = 1 if mode == "per_clip" else min(mix, n_clips - done)
        pool = torch.cat([triplets(next(clips)) for _ in range(k)])
        done += k
        if mode != "per_clip":
            pool = pool[torch.randperm(pool.shape[0], generator=gen)]
        step = pool.shape[0] if mode == "per_clip" else batch
        for s in range(0, pool.shape[0], step):
            _residual(student, pool[s:s + step].to(device))
    for m in student.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1


def _y(blk: nn.Module, clip: torch.Tensor) -> torch.Tensor:
    from models.shared_temporal import triplets
    x = triplets(clip)
    return x[:, 3:6] - _residual(blk, x)


if __name__ == "__main__":
    main()
