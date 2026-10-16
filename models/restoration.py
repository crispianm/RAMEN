"""
Degradation-conditioned restoration modules for D3v2
=====================================================

Three lightweight modules that make D3 robust to low-light/noise/blur degradation
while keeping the DA3 backbone frozen:

  CurveFrontEnd               — Zero-DCE-style learnable illumination correction applied
                                to frames in [0, 1] BEFORE ImageNet normalisation. Pulls
                                degraded inputs back into the backbone's input
                                distribution. Trained with a pixel L1 against the paired
                                clean frames (no gradients through the frozen backbone).
  DegradationProfileEstimator — tiny CNN that predicts the degradation profile
                                (s, gamma, gain, sigma_read, blur_sigma) from the
                                degraded clip; supervised for free by the synthesis
                                parameters. Its embedding FiLM-conditions restoration.
  RestorationMixer            — temporal-attention token restoration at a backbone tap:
                                noise is i.i.d. per frame while scene content is
                                temporally coherent, so attending over time is the right
                                prior for restoring degraded tokens. FiLM-conditioned on
                                the degradation embedding; zero-init residuals → exact
                                identity at init (D3's invariant).

All modules are zero-init where they touch the main path, so a D3v2 model at init
reproduces the current D3 output exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeblurFrontEnd(nn.Module):
    """Spatial deblur+denoise frontend: shallow UNet at 1/2 resolution.

    Predicts an additive residual correction to the input frame. Spatial
    convolutions give it the receptive field needed to undo Gaussian blur
    (σ=2.5 at full res ≈ σ=1.25 at 1/2 res — visible to a 3-level UNet).
    Zero-init output → identity at init. ~200K params.
    """

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(ch * 4 + ch * 2, ch * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch * 2 + ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames`` ``(B, C, T, H, W)`` in [0, 1] → restored, same shape."""
        B, C, T, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).clamp(0.0, 1.0)
        # Work at 1/2 resolution: σ_blur 2.5 → ~1.25 at this scale, visible to 3×3 kernels
        half = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        e1 = self.enc1(half)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="nearest"), e1], 1))
        res = F.interpolate(self.out(d1), size=(H, W), mode="bilinear", align_corners=False)
        return (x + res).clamp(0.0, 1.0).reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


class InputAdapter(DeblurFrontEnd):
    """DN-01 PA (user-approved 2026-09-24): learned PRE-ENCODER input adapter, ``x' = x + R(x)`` on the [0,1] frames
    before the frozen encoder's 14x14 patch embed. R is DeblurFrontEnd's UNet on the pixel-unshuffled frame (3k^2
    channels at 1/k, reflect-padded to a multiple of k) with a zero-init 3k^2-channel output pixel-shuffled back (the
    restoration head's unet_shuffle path without features), so every pixel gets its own prediction and step 0 is x.
    No output clamp: clamp(x) != x on VDA's bicubic-resize overshoot (H4), which would break the step-0 identity."""

    def __init__(self, ch: int = 16, k: int = 4) -> None:
        super().__init__(ch)
        self.k = k
        self.enc1[0] = nn.Conv2d(3 * k * k, ch, 3, padding=1)
        self.out = nn.Conv2d(ch, 3 * k * k, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``(..., 3, H, W)`` in [0, 1] -> adapted, same shape."""
        shape, k = x.shape, self.k
        H, W = shape[-2:]
        x = x.reshape(-1, *shape[-3:])
        ph, pw = -H % k, -W % k
        e1 = self.enc1(F.pixel_unshuffle(F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x, k))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="nearest"), e1], 1))
        return (x + F.pixel_shuffle(self.out(d1), k)[..., :H, :W]).reshape(shape)


def _keep_input(module: nn.Module, inp) -> None:
    # Module-level (not a closure) so deepcopy'd front-ends capture on their own copy.
    module._ramen_in = inp[0]


POCKET_CHECKPOINTS = ("/mnt/DATA/DegradedDepth/RAMEN/checkpoints/pocketdvdnet_v1.0.pt",    # crispwork
                      "/projects/b5dh/benchmark/weights/pocketdvdnet_v1.0.pt")             # Isambard


class PocketFrontEnd(nn.Module):
    """DN-01 FR (user 2026-09-24): the pretrained PocketDVDNet (vendor/pocketdvdnet: blind, 5-frame window -> centre
    frame, v1.0 checkpoint) as a CLIP-level video denoiser in front of VDA-S. The one implementation behind both the
    untouched cascade (bench_joint ``pocketdvdnet``) and the end-to-end FR model, so FR's step 0 IS the cascade.
    Clip ``(N,3,H,W)`` in [0,1] -> denoised, clamped: each frame's t-2..t+2 neighbourhood reflected at the clip ends
    (bench_joint.fastdvdnet_denoise's convention), reflect-padded to a multiple of 4, frames in chunks of ``batch``.
    BatchNorm stays in eval mode under train(): fine-tuning moves weights only (running statistics frozen), so
    train- and eval-mode forwards agree.
    DN-01 FR2 (``depth_branch``): a zero-init branch r on temp2's last features (x0 + x1, the 16-ch full-res
    input of temp2.output_block, captured by a forward pre-hook so vendor/pocketdvdnet stays verbatim);
    ``forward(x, True)`` returns (y, r), r = 0 exactly at init. ``trunk_grad`` beta scales the gradient r sends back into
    the shared trunk (1 = FR2 as run, 0 = trunk stop-grad); the branch's own gradient and every forward value are unchanged."""

    def __init__(self, checkpoint: str | None = None, batch: int = 8, depth_branch: bool = False,
                 trunk_grad: float = 1.0) -> None:
        super().__init__()
        from pathlib import Path

        from vendor.pocketdvdnet import PocketDVDNet, load_pocketdvdnet_checkpoint
        self.checkpoint = checkpoint or next((p for p in POCKET_CHECKPOINTS if Path(p).is_file()), None)
        if self.checkpoint is None:
            raise FileNotFoundError(f"PocketDVDNet v1.0 checkpoint not found in {POCKET_CHECKPOINTS}")
        self.net = PocketDVDNet()
        load_pocketdvdnet_checkpoint(self.net, self.checkpoint, map_location="cpu")
        self.batch = batch
        self.trunk_grad = float(trunk_grad)
        self.branch = None
        if depth_branch:
            c = self.net.recipe.stem_channels
            self.branch = nn.Sequential(nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 3, 3, padding=1))
            nn.init.zeros_(self.branch[-1].weight)
            nn.init.zeros_(self.branch[-1].bias)
            self.net.temp2.output_block.register_forward_pre_hook(_keep_input)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x: torch.Tensor, with_branch: bool = False):
        n, _, h, w = x.shape
        ph, pw = -h % 4, -w % 4
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x
        idx = (torch.arange(n, device=x.device)[:, None] + torch.arange(-2, 3, device=x.device)[None]).abs()
        idx = torch.where(idx > n - 1, 2 * (n - 1) - idx, idx).clamp(0, n - 1)
        ys, rs, ob = [], [], self.net.temp2.output_block
        for s in range(0, n, self.batch):
            ys.append(self.net(xp[idx[s:s + self.batch]].flatten(1, 2)).clamp(0, 1)[..., :h, :w])
            if self.branch is not None:
                feat, ob._ramen_in = ob._ramen_in, None     # never left on the module (deepcopy / memory)
                if with_branch:
                    if self.trunk_grad != 1.0:                   # value unchanged, d/dfeat scaled by beta
                        feat = feat.detach() + self.trunk_grad * (feat - feat.detach())
                    rs.append(self.branch(feat)[..., :h, :w])
        return (torch.cat(ys), torch.cat(rs)) if with_branch else torch.cat(ys)


class CurveFrontEnd(nn.Module):
    """Zero-DCE-style iterative curve enhancement (``LE(x) = x + a·x·(1−x)`` × K).

    A small CNN at 1/4 resolution predicts ``K`` per-pixel, per-channel curve maps that
    are upsampled and applied iteratively. The last conv is zero-init → ``a = 0`` →
    identity at init. ~20k params, negligible latency.
    """

    def __init__(self, hidden: int = 16, iters: int = 3) -> None:
        super().__init__()
        self.iters = iters
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3 * iters, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)  # identity at init
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames`` ``(B, C, T, H, W)`` in [0, 1] → enhanced, same shape/range."""
        B, C, T, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).clamp(0.0, 1.0)
        small = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        a = F.interpolate(self.net(small), size=(H, W), mode="bilinear", align_corners=False)
        out = x
        for k in range(self.iters):
            ak = a[:, 3 * k:3 * (k + 1)]
            out = (out + ak * out * (1.0 - out)).clamp(0.0, 1.0)
        return out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


class WienerDeconv(nn.Module):
    """DN-01 R1 (docs/DN01_FIX_PROPOSALS.md): blind Wiener deconvolution AFTER the restoration head, i.e.
    FastDVDnet -> Wiener's denoise -> deconvolve order inside one network. An MLP on pooled, group-normed
    refinenet1 features estimates each clip's (blur sigma, noise sigma), supervised by the loader's
    ``degrade_params`` (:meth:`targets`), and the frame is deconvolved in closed form with that Gaussian's OTF:

        W(f) = H(f) (1 + lam) / (H(f)^2 + lam),   H(f) = exp(-2 pi^2 sigma_b^2 |f|^2),   lam = exp(log_k + beta t_n)

    (the ``(1 + lam)`` makes the DC gain 1, so sigma_b = 0 is the identity; more estimated noise -> larger lam ->
    less sharpening). ``out = x + alpha (wiener(x) - x)`` with alpha zero-init, so step 0 is the identity.
    ponytail: scalar NSR; a learned (anisotropic) noise spectrum is the upgrade for row-correlated sensor noise.
    The FFT runs in fp32 with a reflect pad of 10 px (>= 3 sigma at the 3 px training maximum)."""

    BLUR_MAX, NOISE_MAX, PAD = 3.0, 50.0 / 255.0, 10    # awgn_blur's ranges: sigma_b <= 3 px, sigma_n <= 50/255

    def __init__(self, in_ch: int, hidden: int = 64) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, in_ch)
        self.est = nn.Sequential(nn.Linear(in_ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 2))
        nn.init.zeros_(self.est[-1].weight)
        nn.init.constant_(self.est[-1].bias, -4.0)       # softplus(-4) = 0.018: estimates start near 0
        self.log_k = nn.Parameter(torch.tensor(math.log(0.01)))   # bench_joint's cascade K
        self.beta = nn.Parameter(torch.zeros(()))
        self.alpha = nn.Parameter(torch.zeros(()))

    @classmethod
    def targets(cls, degrade_params: torch.Tensor) -> torch.Tensor:
        """Loader ``degrade_params`` (B,5) = [s, gamma, gain, sigma_n, sigma_b px] -> normalised (B,2) [t_b, t_n].
        sigma_b < 1/6 px is labelled 0: _blur_frames' kernel is then 1 px (a silent identity; blur bug hunt 2026-09-24)."""
        sb = torch.where(degrade_params[:, 4] < 1 / 6, torch.zeros_like(degrade_params[:, 4]), degrade_params[:, 4])
        return torch.stack([sb / cls.BLUR_MAX, degrade_params[:, 3] / cls.NOISE_MAX], 1)

    def estimate(self, feat: torch.Tensor, clip_len: int) -> torch.Tensor:
        """``feat`` (B*T,C,h,w) -> (B*T,2) normalised [t_b, t_n], averaged over each clip (one profile per clip)."""
        t = F.softplus(self.est(self.norm(feat).mean((2, 3)).float()))
        return t.view(-1, clip_len, 2).mean(1, keepdim=True).expand(-1, clip_len, 2).reshape(-1, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``x`` (N,3,H,W) in [0,1], ``t`` (N,2) from :meth:`estimate` -> deconvolved, same shape."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            p = self.PAD
            xp = F.pad(xf, (p, p, p, p), mode="reflect")
            hp, wp = xp.shape[-2:]
            fy = torch.fft.fftfreq(hp, device=x.device)[:, None]
            fx = torch.fft.rfftfreq(wp, device=x.device)[None, :]
            sb = (t[:, 0] * self.BLUR_MAX).view(-1, 1, 1, 1)
            lam = torch.exp(self.log_k + self.beta * t[:, 1]).view(-1, 1, 1, 1)
            h = torch.exp(-2 * math.pi ** 2 * sb ** 2 * (fx ** 2 + fy ** 2))
            w = h * (1 + lam) / (h ** 2 + lam)
            y = torch.fft.irfft2(torch.fft.rfft2(xp) * w, s=(hp, wp))[..., p:-p, p:-p]
            out = (xf + self.alpha * (y - xf)).clamp(0.0, 1.0)
        return out.to(x.dtype)


class PixelWiener(nn.Module):
    """DN-01 FR+W (Manager 2026-09-24, after the blur bug hunt): blind Wiener deconvolution of the front-end's output
    BEFORE the encoder, with a PIXEL-based per-clip (sigma_b, sigma_n) estimate (R1's estimator read refinenet1, which is
    after the encoder -- circular here). A small full-resolution conv trunk on the frames feeds R1's WienerDeconv
    (Gaussian OTF, DC gain 1, alpha zero-init -> step 0 is the identity); ``last_est`` is supervised by degrade_params
    through train.py's ``restoration_est_weight``. The bug hunt: with sigma known, a small filter recovers ~+2-3 dB at b3s0."""

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.wiener = WienerDeconv(2 * ch)
        self.last_est = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One clip ``(T,3,H,W)`` in [0,1] -> deconvolved; the estimate is pooled over the clip."""
        self.last_est = self.wiener.estimate(self.trunk(x), x.shape[0])
        return self.wiener(x, self.last_est)


class RestorationHead(nn.Module):
    """JR-01 / DN-01: residual RGB restoration read off the depth decoder's post-temporal
    features (the DPT ``refinenet1`` output), so it shares the encoder, the DPT decoder and
    the temporal attention with depth. Zero-init last conv -> ``restored == rgb`` at init.

    DN-01 Phase B (speed): ``hidden`` is the width; ``fuse_scale < 1`` runs the fuse convs at
    that fraction of the frame resolution and bilinearly upsamples the residual (same
    parameters, so the state dict is identical in shape to the full-res head's)."""

    def __init__(self, in_ch: int, hidden: int = 32, fuse_scale: float = 1.0, feat_norm: str = "none",
                 rgb_frames: int = 1, branch: str = "fuse", unet_ch: int = 32, unet_scale: float = 0.5,
                 unet_shuffle: int = 1, wiener: bool = False) -> None:
        super().__init__()
        self.fuse_scale = float(fuse_scale)
        # DN-01 R1 2026-09-24: optional blind Wiener step after the head (WienerDeconv); False = no extra
        # parameters, identical state dict and output.
        self.wiener = WienerDeconv(in_ch) if wiener else None
        self.last_est = None
        # DN-01 amendment 2026-09-23: a metric depth head's refinenet1 features are ~12x a relative head's
        # (T4: mean|x| 64 vs 5), which swamps the rgb path of fuse[0] (164x vs 16x) and left the head at its
        # zero init. "groupnorm" normalises them first (scale-invariant); "none" keeps the old module
        # exactly (no extra parameters, identical state dict).
        if feat_norm not in ("none", "groupnorm"):
            raise ValueError(f"restoration feat_norm must be 'none' or 'groupnorm', got {feat_norm!r}")
        self.norm = nn.GroupNorm(8, in_ch) if feat_norm == "groupnorm" else None
        # DN-01 diagnostic 2026-09-23: rgb_frames 3 / 5 feeds the fuse the raw t-1..t+1 / t-2..t+2 degraded
        # frames (FastDVDnet's multi-frame pixel access); 1 = the old head exactly. The residual is still added
        # to frame t, and the output conv stays zero-init, so step 0 is identity either way.
        if rgb_frames not in (1, 3, 5):
            raise ValueError(f"restoration rgb_frames must be 1, 3 or 5, got {rgb_frames!r}")
        self.rgb_frames = rgb_frames
        self.feat = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True))
        # DN-01 2026-09-24: the fuse's rgb path sees 5 px, the eval blur kernel at sigma 3 is 19 px
        # (scripts/dn01_loader_diag.py). "unet" swaps the fuse for DeblurFrontEnd's 3-level UNet at `unet_scale`
        # of the frame (rgb receptive field 88 px at 1/2, 174 px at 1/4), fed [feat projection, frames]; its
        # output conv is zero-init, so step 0 is identity. "fuse" = the old head exactly (same state dict).
        if branch not in ("fuse", "unet"):
            raise ValueError(f"restoration branch must be 'fuse' or 'unet', got {branch!r}")
        self.branch = branch
        if branch == "fuse":
            self.fuse = nn.Sequential(nn.Conv2d(hidden + 3 * rgb_frames, hidden, 3, padding=1),
                                      nn.ReLU(inplace=True), nn.Conv2d(hidden, 3, 3, padding=1))
            nn.init.zeros_(self.fuse[-1].weight)
            nn.init.zeros_(self.fuse[-1].bias)
        else:
            # DN-01 2026-09-24 (UNet A/B FAIL): a residual predicted at 1/k and upsampled can remove only ~1/k^2 of
            # white pixel noise. unet_shuffle k > 1 = space-to-depth instead of resampling: pixel-unshuffle the
            # frames by k (3*frames*k^2 channels at 1/k), and pixel-shuffle a 3*k^2-channel output back, so every
            # full-res pixel gets its own prediction; unet_scale is then 1/k. k = 1 = the resampling branch above.
            self.unet_shuffle = int(unet_shuffle)
            k2 = self.unet_shuffle ** 2
            self.unet_scale = 1.0 / self.unet_shuffle if self.unet_shuffle > 1 else float(unet_scale)
            self.unet = DeblurFrontEnd(unet_ch)
            self.unet.enc1[0] = nn.Conv2d(hidden + 3 * rgb_frames * k2, unet_ch, 3, padding=1)
            if self.unet_shuffle > 1:
                self.unet.out = nn.Conv2d(unet_ch, 3 * k2, 3, padding=1)
                nn.init.zeros_(self.unet.out.weight)
                nn.init.zeros_(self.unet.out.bias)

    def _unet(self, feat: torch.Tensor, rgb: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
        u, k = self.unet, self.unet_shuffle
        H, W = rgb.shape[-2:]
        if k > 1:
            ph, pw = -H % k, -W % k                                      # 518 is not a multiple of 4: reflect-pad
            x = F.pixel_unshuffle(F.pad(pix, (0, pw, 0, ph), mode="reflect") if ph or pw else pix, k)
        else:
            # area, not bilinear: bilinear without antialias reads 2 px per sample, so at 1/4 it never sees 3/4 of
            # the input pixels (a pixel 15 px off-centre had no effect at all); area averages every pixel of a block
            x = F.interpolate(pix, scale_factor=self.unet_scale, mode="area")
        f = F.interpolate(self.feat(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
        e1 = u.enc1(torch.cat([f, x.to(f.dtype)], 1))
        e2 = u.enc2(F.avg_pool2d(e1, 2))
        e3 = u.enc3(F.avg_pool2d(e2, 2))
        d2 = u.dec2(torch.cat([F.interpolate(e3, size=e2.shape[2:], mode="nearest"), e2], 1))
        d1 = u.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="nearest"), e1], 1))
        if k > 1:
            res = F.pixel_shuffle(u.out(d1), k)[..., :H, :W]
        else:
            res = F.interpolate(u.out(d1), size=(H, W), mode="bilinear", align_corners=False)
        return (rgb + res).clamp(0.0, 1.0)

    def forward(self, feat: torch.Tensor, rgb: torch.Tensor, ctx: torch.Tensor | None = None,
                clip_len: int | None = None) -> torch.Tensor:
        """``feat`` ``(N, C, h, w)``, ``rgb`` ``(N, 3, H, W)`` in [0, 1] -> restored ``(N, 3, H, W)``.
        ``ctx`` ``(N, 3*rgb_frames, H, W)``: the frames the fuse sees (rgb_frames 3); None = ``rgb``.
        With the R1 Wiener step, ``clip_len`` (T of the N = B*T frames) pools its estimate per clip, and the
        estimate is kept in ``self.last_est`` for train.py's supervision."""
        out = self._restore(feat, rgb, ctx)
        if self.wiener is None:
            return out
        self.last_est = self.wiener.estimate(feat, clip_len or feat.shape[0])
        return self.wiener(out, self.last_est)

    def _restore(self, feat: torch.Tensor, rgb: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
        if self.norm is not None:
            feat = self.norm(feat)
        pix = rgb if ctx is None else ctx
        if pix.shape[1] != 3 * self.rgb_frames:
            raise ValueError(f"fuse expects {3 * self.rgb_frames} rgb channels, got {pix.shape[1]}")
        if self.branch == "unet":
            return self._unet(feat, rgb, pix)
        if self.fuse_scale != 1.0:
            small = F.interpolate(pix, scale_factor=self.fuse_scale, mode="bilinear", align_corners=False)
            f = F.interpolate(self.feat(feat), size=small.shape[-2:], mode="bilinear", align_corners=False)
            res = self.fuse(torch.cat([f, small.to(f.dtype)], 1))
            res = F.interpolate(res, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            return (rgb + res).clamp(0.0, 1.0)
        f = F.interpolate(self.feat(feat), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return (rgb + self.fuse(torch.cat([f, pix.to(f.dtype)], 1))).clamp(0.0, 1.0)


def neighbour_stack(frames: torch.Tensor, k: int = 3) -> torch.Tensor:
    """``(B, T, 3, H, W)`` -> ``(B*T, 3k, H, W)``: frames t-k//2 .. t+k//2 per t, reflected at the clip
    ends (as FastDVDnet does: t-1 of the first frame is frame 1) and clamped for clips shorter than k."""
    T = frames.shape[1]
    t = torch.arange(T, device=frames.device)
    parts = []
    for o in range(-(k // 2), k // 2 + 1):
        i = (t + o).abs()                                   # reflect below 0
        i = torch.where(i > T - 1, 2 * (T - 1) - i, i)      # reflect above T-1
        parts.append(frames[:, i.clamp(0, T - 1)])
    return torch.cat(parts, 2).flatten(0, 1)


class DegradationProfileEstimator(nn.Module):
    """Predict the degradation profile of a clip and embed it for FiLM conditioning.

    Returns ``(e, params)``: ``e`` ``(B, cond_dim)`` conditioning embedding (mean over
    frames), ``params`` ``(B, 5)`` predicted ``[s, gamma, gain, sigma_read, blur_sigma]``
    — supervised by the true synthesis parameters (`degrade_params` in the batch).
    ~0.1M params; runs on 1/4-resolution frames.
    """

    def __init__(self, cond_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, cond_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(cond_dim, 5)

    def forward(self, frames: torch.Tensor):
        """``frames`` ``(B, C, T, H, W)`` in [0, 1] (RAW degraded input, pre-frontend)."""
        B, C, T, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        e = self.features(x).flatten(1).reshape(B, T, -1).mean(1)  # (B, cond_dim)
        return e, self.head(e)


class CausalDegradationProfileEstimator(nn.Module):
    """Per-frame causal counterpart of :class:`DegradationProfileEstimator` (EXPT-08).

    The base estimator's ``e`` is a mean over the WHOLE clip, including frames a
    per-frame router at time ``t`` has not seen yet -- routing on it would be
    non-causal (breaks C2) despite looking like an online estimate. This class
    instead computes, per frame, the RUNNING mean of the same per-frame features
    over frames ``0..t`` only -- prefix-invariant and causal by construction, the
    same pattern :class:`RestorationMixer` already uses (see
    ``tests/test_causal.py``'s ``test_restoration_is_prefix_invariant`` /
    ``test_restoration_ignores_future_perturbations``).

    Shares the exact same per-frame feature extractor and head as the base class
    (not reimplemented) so a causal and whole-clip estimate only ever differ in
    the pooling step, never in what each frame's features encode.

    Returns ``(e, params)``: ``e`` ``(B, T, cond_dim)`` -- ``e[:, t]`` is the causal
    running-mean embedding using only frames ``0..t``; ``params`` ``(B, T, 5)`` the
    per-frame predicted profile from that embedding (supervised against the clip's
    single true synthesis profile, broadcast across T -- this repo's degradation is
    one profile per clip, not per frame, so there is no finer-grained truth to
    supervise against; the causal constraint is about what the ROUTER may see per
    frame, not about predicting a profile that itself varies within a clip).
    """

    def __init__(self, cond_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, cond_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(cond_dim, 5)

    def forward(self, frames: torch.Tensor):
        """``frames`` ``(B, C, T, H, W)`` in [0, 1] (RAW degraded input, pre-frontend)."""
        B, C, T, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        f = self.features(x).flatten(1).reshape(B, T, -1)          # (B, T, cond_dim) per-frame
        counts = torch.arange(1, T + 1, device=f.device, dtype=f.dtype).view(1, T, 1)
        e = f.cumsum(dim=1) / counts                                # (B, T, cond_dim) running mean
        return e, self.head(e)


class _FiLM(nn.Module):
    """Zero-init FiLM: ``h * (1 + g(e)) + b(e)`` — identity when the embedding is unused."""

    def __init__(self, cond_dim: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, e: torch.Tensor | None) -> torch.Tensor:
        if e is None:
            return h
        g, b = self.proj(e).chunk(2, dim=-1)          # (B, dim) each
        # h is (B*N, T, dim): expand the per-clip modulation over tokens and time.
        n = h.shape[0] // e.shape[0]
        g = g.repeat_interleave(n, 0).unsqueeze(1)
        b = b.repeat_interleave(n, 0).unsqueeze(1)
        return h * (1.0 + g) + b


class RestorationMixer(nn.Module):
    """Degradation-conditioned temporal token restoration (one backbone tap).

    Block = [FiLM → temporal self-attention over T] + [FiLM → ConvFFN with a depthwise
    3×3 spatial conv]. Both residual branches are zero-init at their output projection,
    so the block is the exact identity at init. ``needs_cond`` marks it for the
    conditioning plumbing in ``D3Net.decode_modules``.
    """

    needs_cond = True

    def __init__(self, dim: int = 768, temporal_kernel: int = 3, num_heads: int = 8,
                 cond_dim: int = 128, causal: bool = True) -> None:
        super().__init__()
        del temporal_kernel  # signature-compatible with the other mixers
        self.causal = bool(causal)
        self.norm1 = nn.LayerNorm(dim)
        self.film1 = _FiLM(cond_dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)      # identity at init
        nn.init.zeros_(self.attn.out_proj.bias)

        self.norm2 = nn.LayerNorm(dim)
        self.film2 = _FiLM(cond_dim, dim)
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.dw = nn.Conv2d(2 * dim, 2 * dim, 3, padding=1, groups=2 * dim)  # spatial mixing
        self.fc2 = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.fc2.weight)                # identity at init
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor, ph: int, pw: int,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, C = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(B * N, T, C)   # attend over time per position

        h = self.film1(self.norm1(x), cond)
        # Strictly causal temporal restoration.  A frame may attend to itself and
        # earlier frames, never to a later frame.  Besides making batched training
        # honest, this makes a prefix forward equivalent to online inference.
        causal_mask = (torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
                       if self.causal else None)
        a, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + a

        h = self.film2(self.norm2(x), cond)
        h = F.gelu(self.fc1(h))                                # (B*N, T, 2C)
        g = h.reshape(B, N, T, -1).permute(0, 2, 3, 1).reshape(B * T, -1, ph, pw)
        g = F.gelu(self.dw(g))                                 # depthwise 3×3 over the token grid
        h = g.reshape(B, T, -1, N).permute(0, 3, 1, 2).reshape(B * N, T, -1)
        x = x + self.fc2(h)

        return x.reshape(B, N, T, C).permute(0, 2, 1, 3)
