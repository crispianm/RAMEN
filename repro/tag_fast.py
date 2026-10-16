"""EXPLORATORY, UNAUDITED. Device-resident implementations of the TAG (adaptive-gain EMA, ``agf``) recursion of evaluation/depth_probe.depth_agf.

Same maths as the reference (acc_0 = d_0; u = (d - acc) / max(|acc|, 1e-6); s = beta s + (1-beta) u^2; a = amin + (1-amin)(1 - exp(-s/tau^2)); acc += a (d - acc)); only the arithmetic differs:
  V1  evaluation.depth_probe.depth_agf_torch   (fp64 torch loop; not in this file)
  V2  agf_v2    fp32 device-resident torch loop, 1/tau^2 folded to a constant, one reciprocal per pixel
  V3  agf_v3    one fused Triton kernel over the whole (N,H,W) window (per-pixel state in registers)
  V3s Agf3Stream.step(d_t)  the same update as ONE frame step with carried (acc, s); bit-identical to agf_v3 (shared ``_update``)
Input may be fp32 or fp16 (cast to fp32 on load); output is always fp32. amin == 1 returns a copy of the input (bit-exact identity, as the reference)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

AGF_BETA = 0.5
_BLOCK = 256


@torch.no_grad()
def agf_v2(x: torch.Tensor, amin: float, tau: float, beta: float = AGF_BETA) -> torch.Tensor:
    if amin == 1.0:
        return x.float().clone()
    k = 1.0 / (tau * tau)
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    acc = x[0].float()
    out[0] = acc
    s = torch.zeros_like(acc)
    for t in range(1, x.shape[0]):
        d = x[t].float()
        u = (d - acc) * torch.reciprocal(acc.abs().clamp_min(1e-6))
        s = beta * s + (1.0 - beta) * u * u
        a = amin + (1.0 - amin) * (1.0 - torch.exp(-(s * k)))
        acc = acc + a * (d - acc)
        out[t] = acc
    return out


@triton.jit
def _update(acc, s, d, amin, k, beta):
    u = (d - acc) / tl.maximum(tl.abs(acc), 1e-6)
    s = beta * s + (1.0 - beta) * u * u
    a = amin + (1.0 - amin) * (1.0 - tl.exp(-(s * k)))
    return acc + a * (d - acc), s


@triton.jit
def _window_kernel(x_ptr, o_ptr, N, P, amin, k, beta, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < P
    acc = tl.load(x_ptr + offs, mask=m, other=1.0).to(tl.float32)
    tl.store(o_ptr + offs, acc, mask=m)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(1, N):
        d = tl.load(x_ptr + t * P + offs, mask=m, other=1.0).to(tl.float32)
        acc, s = _update(acc, s, d, amin, k, beta)
        tl.store(o_ptr + t * P + offs, acc, mask=m)


@triton.jit
def _step_kernel(d_ptr, acc_ptr, s_ptr, o_ptr, P, amin, k, beta, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < P
    acc = tl.load(acc_ptr + offs, mask=m, other=1.0)
    s = tl.load(s_ptr + offs, mask=m, other=0.0)
    d = tl.load(d_ptr + offs, mask=m, other=1.0).to(tl.float32)
    acc, s = _update(acc, s, d, amin, k, beta)
    tl.store(acc_ptr + offs, acc, mask=m)
    tl.store(s_ptr + offs, s, mask=m)
    tl.store(o_ptr + offs, acc, mask=m)


@torch.no_grad()
def agf_v3(x: torch.Tensor, amin: float, tau: float, beta: float = AGF_BETA) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 3
    if amin == 1.0:
        return x.float().clone()
    x = x.contiguous()
    o = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    P = x[0].numel()
    _window_kernel[(triton.cdiv(P, _BLOCK),)](x, o, x.shape[0], P, float(amin), 1.0 / (tau * tau), float(beta), BLOCK=_BLOCK)
    return o


class Agf3Stream:
    """Causal one-frame-at-a-time TAG: ``first`` = frame 0 (returned unchanged as fp32), then ``step(d_t)`` per frame with carried (acc, s)."""

    def __init__(self, first: torch.Tensor, amin: float, tau: float, beta: float = AGF_BETA):
        self.amin, self.k, self.beta = float(amin), 1.0 / (tau * tau), float(beta)
        self.acc = first.float().clone().contiguous()
        self.s = torch.zeros_like(self.acc)
        self.P = self.acc.numel()

    @torch.no_grad()
    def step(self, d: torch.Tensor) -> torch.Tensor:
        if self.amin == 1.0:
            return d.float().clone()
        o = torch.empty_like(self.acc)
        _step_kernel[(triton.cdiv(self.P, _BLOCK),)](d.contiguous(), self.acc, self.s, o, self.P, self.amin, self.k, self.beta, BLOCK=_BLOCK)
        return o

    @torch.no_grad()
    def chunk(self, x: torch.Tensor) -> torch.Tensor:
        """Filter the next ``(n,H,W)`` chunk of the stitched sequence (not its first frame) with carried state."""
        return torch.stack([self.step(x[i]) for i in range(x.shape[0])])
