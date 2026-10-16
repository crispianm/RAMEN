"""DN-01 F3 inference-only baselines (scoring only, DEV only). Read PRE-DECLARED by the Auditor: audit/F3_predeclared_20261001.md (sha256 637d269f...1145), the tau amendment
audit/F3_amendment_tau_20261001.md (sha256 b4650d22...6a51) and the reporting addendum audit/F1E6_reporting_addendum_20261001.md (sha256 a60fd8e3...). Nothing here trains a depth model.
NO upstream code is vendored: each baseline's own released file is imported from a pinned clone (bench_joint.PATHS; the clones and weights are fetched by
provenance/dn01_f3/submit_f3.sh STAGE=fetch and sha-checked), so the licences stay with the upstream repos (Zero-DCE++: CC BY-NC 4.0; DVP: no licence file, research code).

(i)  ENHANCEMENT CASCADE ``zerodcepp+metric_vda_s``: Zero-DCE++ (Li, Guo, Loy, TPAMI 2021; github.com/Li-Chongyi/Zero-DCE_extension @09f202b6, Zero-DCE++/model.py
     ``enhance_net_nopool``, released weights Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth sha256 ca8855b9...3b84) as a bench_joint pre-restorer feeding released Metric-VDA-S.
     Repo default inference (lowlight_test.py): scale_factor 12, RGB in [0,1], output = the enhanced image (saved through torchvision.save_image = clamped to [0,1]).
     Deviations, forced by the harness, NOT tuning: the repo CROPS each image to a multiple of 12; here the frame is replicate-padded to a multiple of 12 and the output cropped back
     (the scorer needs every pixel); no 8-bit save/re-load between the enhancer and the depth model; frames in batches of 8 (the network has no batch coupling). Blind (ignores sigma / blur).
(iii) BLIND TEMPORAL-CONSISTENCY POST-PROCESS ``dvp``: Deep Video Prior (Lei, Xing, Chen, NeurIPS 2020), the authors' official PyTorch implementation
     github.com/yzxing87/pytorch-deep-video-prior @0efb8aeb (models/network.py ``UNet``, main_IRT.py's unimodal defaults: --with_IRT 0, --max_epoch 25, perceptual loss = L1 on
     ImageNet-normalised images + the VGG19 conv1_2..conv5_2 L1 terms with weights 1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5 on torchvision's ImageNet VGG19 (vgg19-dcbb9e9d.pth), Adam lr 1e-4,
     xavier-normal init with zero bias, seed 2020, frames visited in order once per epoch, output = the net's prediction for every frame after epoch 25). DVP has NO pretrained weights: it is
     test-time training per video, so it is NON-CAUSAL (frame t's output depends on every frame of the clip) and costs minutes per clip (wall-clock printed per clip in the job log).
     It acts where every depth post token acts (bench_joint.run_pipeline: the stitched depth at the scorer's size). Its "input video" = the depth model's input frames (guide ``i``,
     resized bilinearly to the depth size); its "processed video" = the depth video, mapped to an image by a per-clip log-depth min/max ([log max(min d, 1e-3), log max d] -> [0, 1],
     replicated to 3 channels; inverted exactly after the net, channel mean, clamped to [0, 1] as the repo's saved images are). Deviations: the repo CROPS to a multiple of 32, here
     reflect-pad and crop back; no 8-bit image round trip. The Lai et al. ECCV 2018 blind-consistency weights were the first choice (causal) but their only host
     (vllab.ucmerced.edu/.../ECCV18_blind_consistency.pth) returns 404 (checked 2026-10-01), so DVP is the named method.
PRE-RULINGS (audit/F3_prerulings_20261001.md, sha256 d2fd18c2...): DVP ACCEPTED (Lai ECCV18 dropped: weights 404), disclosed beside every DVP row as NON-causal, per-clip
     test-time training (its time is per-clip TRAINING time, not comparable to a causal stage), ONE seed (2020); run as an EXTERNAL pinned clone only, never vendored / redistributed.
     The depth map: per-clip log min / max over ALL pixels of the clip (sky and outlier depths included: a fixed rule, not tuned to depth), exact inverse (tested bit-exact on float32
     depth >= 1e-3 m; depths below 1e-3 m are floored there and COUNTED). The net's outputs outside [0, 1] are CLAMPED and COUNTED; every DVP clip records
     ``dvp_out_of_range_px``, ``dvp_floored_px``, ``dvp_px`` and its wall-clock ``dvp_seconds`` in the per-clip results (bench_joint.variant_extras), so the reader prints them per clip / cell.
STAGE RULE (tau amendment): agf's innovation is RELATIVE (tau 0.06 = 6%), so a metric or up-to-scale network takes the stage DIRECTLY; only an affine-ambiguous (free shift) output
would need the oracle-scaled route. Every F3 baseline here is METRIC (Metric-VDA-S, Metric-VDA-L, Zero-DCE++ -> Metric-VDA-S, DVP on Metric-VDA-S): direct, settings unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ZERODCE_SCALE = 12                                   # Zero-DCE++/lowlight_test.py: scale_factor = 12
ZERODCE_WEIGHTS = "Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
ZERODCE_MODEL = "Zero-DCE++/model.py"
DVP_EPOCHS, DVP_SEED, DVP_LR, DVP_FEATURES, DVP_MULT = 25, 2020, 1e-4, 32, 32
DVP_LAYERS = {"conv1_2": 1.0 / 2.6, "conv2_2": 1.0 / 4.8, "conv3_2": 1.0 / 3.7, "conv4_2": 1.0 / 5.6, "conv5_2": 10.0 / 1.5}
DVP_SLICES = ((0, 4), (4, 9), (9, 14), (14, 23), (23, 32))          # pytorch-deep-video-prior/vgg.py: torchvision vgg19().features slices -> conv1_2 .. conv5_2
DEPTH_FLOOR = 1e-3
_IMNET_MEAN = (0.485, 0.456, 0.406)
_IMNET_STD = (0.229, 0.224, 0.225)


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- (i) Zero-DCE++

def build_zerodce(repo: str | Path, device: str) -> nn.Module:
    """The released Zero-DCE++ network (strict load of Epoch99.pth) from the pinned clone."""
    repo = Path(repo)
    net = _load_file("zerodcepp_model", repo / ZERODCE_MODEL).enhance_net_nopool(ZERODCE_SCALE)
    net.load_state_dict(torch.load(repo / ZERODCE_WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    return net.to(device).eval()


@torch.no_grad()
def zerodce_enhance(net: nn.Module, x: torch.Tensor, batch: int = 8) -> torch.Tensor:
    """(T,3,H,W) [0,1] -> enhanced (T,3,H,W) [0,1]: replicate-pad to a multiple of 12, the repo's forward, crop back, clamp (= save_image's clamp)."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % ZERODCE_SCALE, (-w) % ZERODCE_SCALE
    out = []
    for s in range(0, x.shape[0], batch):
        xb = x[s:s + batch].float()
        if ph or pw:
            xb = F.pad(xb, (0, pw, 0, ph), mode="replicate")
        y, _ = net(xb)
        out.append(y[..., :h, :w].clamp(0.0, 1.0).to(x.dtype))
    return torch.cat(out)


# ----------------------------------------------------------------------------- (iii) Deep Video Prior

class _VGG19Feats(nn.Module):
    """pytorch-deep-video-prior/vgg.py's VGG19 wrapper, by its slice indices, on torchvision's ImageNet VGG19 (frozen)."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        import torchvision
        if weights_path:
            feats = torchvision.models.vgg19(weights=None).features
            feats.load_state_dict({k[len("features."):]: v for k, v in torch.load(weights_path, map_location="cpu", weights_only=True).items()
                                   if k.startswith("features.")})
        else:
            feats = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList(nn.Sequential(*[feats[i] for i in range(a, b)]) for a, b in DVP_SLICES)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        out = []
        for s in self.slices:
            x = s(x)
            out.append(x)
        return out


def _imnet(x: torch.Tensor) -> torch.Tensor:
    return (x - x.new_tensor(_IMNET_MEAN).view(1, 3, 1, 1)) / x.new_tensor(_IMNET_STD).view(1, 3, 1, 1)


def dvp_loss(vgg: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """main_IRT.py Lp_loss: L1 on the ImageNet-normalised images + sum_l w_l L1(VGG_l(pred), VGG_l(target))."""
    a, b = _imnet(pred), _imnet(target)
    loss = (a - b).abs().mean()
    for (_, wgt), fa, fb in zip(DVP_LAYERS.items(), vgg(a), vgg(b)):
        loss = loss + wgt * (fa - fb).abs().mean()
    return loss


def _xavier(model: nn.Module) -> None:
    """main_IRT.py initialize_weights."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


def depth_to_image(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(N,H,W) metres -> (N,H,W) in [0,1] by the per-clip log-depth min/max; returns (u, lo, span)."""
    ld = np.log(np.maximum(depth.astype(np.float64), DEPTH_FLOOR))
    lo, hi = float(ld.min()), float(ld.max())
    span = max(hi - lo, 1e-6)
    return (ld - lo) / span, lo, span


def image_to_depth(u: np.ndarray, lo: float, span: float, dtype) -> np.ndarray:
    """The exact inverse of depth_to_image on [0, 1]; values outside are CLAMPED (counted by dvp_filter)."""
    return np.exp(lo + np.clip(u, 0.0, 1.0) * span).astype(dtype)


LAST_STATS: dict = {}       # the stats of the last dvp_filter call (out-of-range / floored pixel counts, wall-clock seconds)


_DVP_NET_CLS = {}
_VGG = {}


def _unet_cls(repo: str | Path):
    key = str(repo)
    if key not in _DVP_NET_CLS:
        _DVP_NET_CLS[key] = _load_file("dvp_network", Path(repo) / "models/network.py").UNet
    return _DVP_NET_CLS[key]


def dvp_filter(depth: np.ndarray, frames: torch.Tensor, repo: str | Path, epochs: int = DVP_EPOCHS, vgg: nn.Module | None = None) -> np.ndarray:
    """Deep Video Prior on one depth video. ``depth`` (N,H,W) metres; ``frames`` (N,3,h,w) [0,1] the depth model's input frames (device = where DVP trains).
    Returns (N,H,W) with depth's dtype. SELF-CONTAINED for autograd (F3 smoke job 6986546: the nuScenes-night scorer's ``score()`` is ``@torch.no_grad``, so the test-time training
    had no graph): the whole training runs under ``torch.inference_mode(False)`` + ``torch.enable_grad()`` on fresh copies of its inputs, whatever the caller's grad mode
    (no_grad, inference_mode or plain), with an identical result (tested)."""
    with torch.inference_mode(False), torch.enable_grad():
        frames = frames.detach().clone()               # an inference-mode tensor cannot enter autograd; a clone made outside inference mode can
        depth = np.array(depth, copy=True)
        return _dvp_train(depth, frames, repo, epochs, vgg)


def _dvp_train(depth: np.ndarray, frames: torch.Tensor, repo: str | Path, epochs: int, vgg: nn.Module | None) -> np.ndarray:
    dev = frames.device
    n, H, W = depth.shape
    t0 = time.time()
    u, lo, span = depth_to_image(depth)
    inp = F.interpolate(frames.float(), size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0) if tuple(frames.shape[-2:]) != (H, W) \
        else frames.float().clamp(0.0, 1.0)
    tgt = torch.from_numpy(u).float().to(dev)[:, None].expand(-1, 3, -1, -1)
    ph, pw = (-H) % DVP_MULT, (-W) % DVP_MULT
    if ph or pw:                                        # the repo crops to a multiple of 32; reflect-pad and crop back instead
        inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        tgt = F.pad(tgt, (0, pw, 0, ph), mode="reflect")
    if vgg is None:
        if str(dev) not in _VGG:                       # built once per device (torchvision ImageNet VGG19 from the torch hub cache)
            _VGG[str(dev)] = _VGG19Feats().to(dev).eval()
        vgg = _VGG[str(dev)]
    torch.manual_seed(DVP_SEED)
    net = _unet_cls(repo)(in_channels=3, out_channels=3, init_features=DVP_FEATURES).to(dev)
    torch.manual_seed(DVP_SEED)
    _xavier(net)
    opt = torch.optim.Adam(net.parameters(), lr=DVP_LR)
    net.train()
    for _ in range(epochs):
        for i in range(n):
            loss = dvp_loss(vgg, net(inp[i:i + 1]), tgt[i:i + 1])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = torch.cat([net(inp[i:i + 1])[..., :H, :W].mean(1) for i in range(n)]).double().cpu().numpy()
    sec = time.time() - t0
    LAST_STATS.clear()
    LAST_STATS.update(dvp_out_of_range_px=int(((out < 0.0) | (out > 1.0)).sum()), dvp_floored_px=int((depth < DEPTH_FLOOR).sum()), dvp_px=int(out.size),
                      dvp_seconds=float(sec))
    print(f"dvp: {n} frames {H}x{W}, {epochs} epochs on {dev}: {sec:.1f} s wall-clock; outputs outside [0,1] clamped: {LAST_STATS['dvp_out_of_range_px']} of "
          f"{out.size} px; depth floored at {DEPTH_FLOOR} m: {LAST_STATS['dvp_floored_px']} px", flush=True)
    return image_to_depth(out, lo, span, depth.dtype)


class _Memo:
    """One DVP run per (clip depth, guide) even when several grid variants start with ``dvp`` (``dvp``, ``dvp+agf0.2t0.06`` ...): a hit needs byte-identical depth and the same guide tensor."""

    def __init__(self):
        self.key, self.out, self.stats = None, None, {}

    @staticmethod
    def make_key(depth: np.ndarray, g: torch.Tensor):
        return (hashlib.sha1(np.ascontiguousarray(depth).tobytes()).hexdigest(), tuple(depth.shape), str(depth.dtype), g.data_ptr(), tuple(g.shape),
                float(g.detach().double().sum()))

    def get(self, depth, g, fn):
        k = self.make_key(depth, g)
        if k != self.key:
            self.key, self.out, self.stats = None, None, {}
            LAST_STATS.clear()
            self.out = fn()
            self.stats = dict(LAST_STATS)
            self.key = k
        return self.out.copy()


MEMO = _Memo()


def make_dvp_post(repo: str | Path):
    """bench_joint depth post token ``dvp`` -> fn(depth, guides) (guide ``i`` = the depth model's input frames)."""
    def fn(depth, guides=None):
        if guides is None or "i" not in guides:
            raise ValueError("dvp needs the 'i' guide frames (run_pipeline provides them)")
        return MEMO.get(depth, guides["i"], lambda: dvp_filter(depth, guides["i"], repo))
    fn.guide = "i"
    return fn
