"""L15 scorer specs (NEW code path only; bench_joint.py is untouched, the pinned released rows keep going through ``ours:``).

  l15s:<ckpt>   SHARED   a plain train.py checkpoint of an L15 fine-tune; depth reads the fine-tuned restorer (= ``ours:`` semantics, restored RGB = the same restorer's R).
  l15d:<packed> DECOUPLED a packed checkpoint (scripts/l15_pack_decoupled.py): depth reads the frozen depth_restorer + the INIT depth weights, the restored RGB is the rgb_restorer's.

Both return the same spec dict as bench_joint._ours' design-(c) branch (fwd -> (depth (T,H,W), restored (T,3,H,W)), window 32), so the scorer's windowing / TAG / metrics code is shared and unchanged."""
from __future__ import annotations

import torch

PREFIXES = ("l15s:", "l15d:")


def load_l15(ckpt: str, device: str, decoupled: bool):
    from models import build_model, trainable_modules
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = dict(ck["cfg"]["model"])
    m["st_temp2_init"] = None                                   # trainable_ema carries all of temp2 (BN buffers incl.); the init path is a training-host path
    m["st_decoupled_restorer"] = bool(decoupled)
    net = build_model({**ck["cfg"], "model": m})
    if decoupled:
        if "rgb_restorer" not in ck and "depth_restorer" not in ck:
            raise ValueError(f"{ckpt} is not a packed decoupled checkpoint (no rgb_restorer / depth_restorer section)")
        net.load_packed(ck)
    else:
        torch.nn.ModuleDict(trainable_modules(net)).load_state_dict(ck.get("trainable_ema", ck.get("trainable")), strict=True)
    return net.to(device).eval()


def ours_l15(device: str, spec: str) -> dict:
    prefix, ckpt = next((p, spec[len(p):]) for p in PREFIXES if spec.startswith(p))
    net = load_l15(ckpt, device, decoupled=(prefix == "l15d:"))

    def fwd(x):                                                  # x: (T,3,H,W)
        rgb = x.permute(1, 0, 2, 3)[None]
        if net.decoupled_restorer:
            r_depth, restored = net.restore_decoupled(rgb)       # depth <- depth_restorer ; output <- rgb_restorer
        else:
            r_depth = restored = net.restore_batch(rgb)
        d, _ = net._depth_and_feats(net.consume(rgb, r_depth))
        return d[0, 0], restored[0].permute(1, 0, 2, 3)
    return dict(fwd=fwd, metric=not net.predicts_disparity, window=32, weights={"ours": ckpt}, net=net)
