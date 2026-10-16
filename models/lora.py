"""
Minimal LoRA for the frozen DA3 backbone — the "naive backbone adaptation" baseline.

Wraps the ``attn.qkv`` linears of the DINOv2 blocks with rank-``r`` adapters
(``W x + (alpha/r)·B A x``, ``B`` zero-init → identity at init). The base weights stay
frozen; only ``A``/``B`` train. Mergeable at deploy time → zero inference cost.

MoLoRA variant (``num_experts > 1``): N expert pairs, routed by a degradation embedding
injected via ``set_cond()`` before each backbone forward. Router is zero-init → uniform
gate at init; all B-matrices are zero-init → identity at init regardless of routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank residual.

    With ``num_experts > 1`` and ``cond_dim > 0``, becomes a Mixture-of-LoRA: N expert
    pairs routed by a ``(B, cond_dim)`` embedding set via :meth:`set_cond` before each
    backbone forward pass.  All B-matrices are zero-init → identity at init.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 num_experts: int = 1, cond_dim: int = 0) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.num_experts = num_experts
        self.lora_a = nn.ModuleList(
            [nn.Linear(base.in_features, rank, bias=False) for _ in range(num_experts)]
        )
        self.lora_b = nn.ModuleList(
            [nn.Linear(rank, base.out_features, bias=False) for _ in range(num_experts)]
        )
        for a in self.lora_a:
            nn.init.kaiming_uniform_(a.weight, a=5 ** 0.5)
        for b in self.lora_b:
            nn.init.zeros_(b.weight)  # identity at init
        # Router: zero-init → uniform gate at init (no routing signal until training)
        if num_experts > 1 and cond_dim > 0:
            self.router = nn.Linear(cond_dim, num_experts, bias=False)
            nn.init.zeros_(self.router.weight)
        else:
            self.router = None
        for p in self.base.parameters():
            p.requires_grad_(False)
        self._cond: torch.Tensor | None = None  # set by set_cond() before backbone forward

    def set_cond(self, cond: torch.Tensor | None) -> None:
        self._cond = cond

    def forward(self, x):
        base_out = self.base(x)
        if self.num_experts == 1:
            lora_out = self.lora_b[0](self.lora_a[0](x)) * self.scale
        else:
            if self.router is not None and self._cond is not None:
                # x: (B*T, ..., in) where leading dim = B*T; cond: (B, cond_dim)
                BT = x.shape[0]
                B = self._cond.shape[0]
                T = max(BT // B, 1)
                c = self._cond.unsqueeze(1).expand(-1, T, -1).reshape(BT, -1)
                g = torch.softmax(self.router(c.to(x.dtype)), dim=-1)  # (BT, E)
            else:
                g = x.new_ones(x.shape[0], self.num_experts) / self.num_experts
            # expert_outs: (BT, E, ..., out); g_view broadcasts over inner dims
            expert_outs = torch.stack(
                [self.lora_b[i](self.lora_a[i](x)) for i in range(self.num_experts)], dim=1
            )
            g_view = g.view(g.shape[0], self.num_experts, *([1] * (x.dim() - 1)))
            lora_out = (expert_outs * g_view).sum(1) * self.scale
        return base_out + lora_out


def inject_lora(module: nn.Module, rank: int = 8, alpha: float = 16.0,
                suffix: str = "qkv", num_experts: int = 1, cond_dim: int = 0) -> list:
    """Replace every ``*.{suffix}`` ``nn.Linear`` under ``module`` with a
    :class:`LoRALinear` (in place). Returns the new wrappers."""
    wrapped = []
    for name, child in list(module.named_modules()):
        for attr, sub in list(child.named_children()):
            if attr == suffix and isinstance(sub, nn.Linear):
                lora = LoRALinear(sub, rank, alpha, num_experts=num_experts, cond_dim=cond_dim)
                setattr(child, attr, lora)
                wrapped.append(lora)
    if not wrapped:
        raise RuntimeError(f"inject_lora: no '*.{suffix}' Linear found under {type(module).__name__}.")
    return wrapped
