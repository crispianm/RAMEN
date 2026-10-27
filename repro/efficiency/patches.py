"""Runtime efficiency patches (monkeypatch/wrapper). NOTHING on disk in the pinned clone / venv is edited.
Class labels (EFFICIENCY task 2026-10-13):
  E = exact reformulation (same maths; fp rounding may differ in the last bits - measured, never assumed bitwise unless stated BITWISE)
  A = approximation / changes output (torch.compile numerics, fp16 vs fp32, ...) - NEVER called the same model.
Variants (flags):
  sdpa      E1   DINOv2 Attention.forward -> F.scaled_dot_product_attention           class E (same softmax(QK^T/sqrt d)V)
  bnfold    E2a  fold eval BatchNorm into the preceding Conv in the restorer          class E (same maths, different rounding)
  cudnn     E2b  torch.backends.cudnn.benchmark=True                                  class E (algorithm choice only)
  chlast    E2c  restorer channels_last                                               class E
  hookoff   E2d  drop the diagnostic refinenet1 collect hook + its torch.cat          class E, BITWISE depth (inject off)
  fp16build E2e  build the stage-1/2 triplet tensors in fp16 (only under autocast), centre frame kept fp32 for the residual
                                                                                   class E, BITWISE (autocast casts conv input to fp16 anyway)
  c_enc/c_head/c_rest  E3  torch.compile(freezing) of encoder / DPT+motion head / restorer blocks   class A
"""
from __future__ import annotations
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

E_FLAGS = ("sdpa", "sdpa_motion", "bnfold", "cudnn", "chlast", "hookoff", "fp16build", "dense_group")
A_FLAGS = ("c_enc", "c_head", "c_rest")


def variant_class(flags) -> str:
    # Auditor 36d0fc14 naming: EXACT = bitwise-only toggles; everything else NEAR (compile included). Earlier JSONs say E/A: E->NEAR (except hookoff/fp16build alone = EXACT), A->NEAR.
    if not flags:
        return "BASE"
    return "EXACT" if set(flags) <= {"hookoff", "fp16build"} else "NEAR"


# ---- E1 ----------------------------------------------------------------------------------------------------------------
def _sdpa_forward(self, x, attn_bias=None):
    assert attn_bias is None
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    o = F.scaled_dot_product_attention(q, k, v)            # default scale = head_dim**-0.5 == self.scale; dropout p=0 (eval)
    o = o.transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(o))


def patch_sdpa():
    from video_depth_anything.dinov2_layers import attention as A
    A.Attention.forward = _sdpa_forward
    A.MemEffAttention.forward = _sdpa_forward              # same fall-back target when xformers is absent


def _motion_attention(self, query, key, value, attention_mask=None):
    assert attention_mask is None and not self.upcast_attention
    o = F.scaled_dot_product_attention(query, key, value, scale=self.scale)     # (B*heads, N, d): 3-D layout of the pinned baddbmm path
    return self.reshape_batch_dim_to_heads(o)


def patch_sdpa_motion():
    from video_depth_anything.motion_module import motion_module as M
    M.TemporalAttention._attention = _motion_attention


@torch.no_grad()
def dense_group(root: nn.Module) -> int:
    """grouped 9->90 (groups=3) input conv -> dense conv with block-diagonal weight (same maths, zeros elsewhere)."""
    n = 0
    for m in root.modules():
        if isinstance(m, nn.Sequential):
            for k, c in list(m._modules.items()):
                if isinstance(c, nn.Conv2d) and c.groups > 1:
                    g, co, ci = c.groups, c.out_channels, c.in_channels
                    w = torch.zeros(co, ci, *c.kernel_size, dtype=c.weight.dtype, device=c.weight.device)
                    for i in range(g):
                        w[i * co // g:(i + 1) * co // g, i * ci // g:(i + 1) * ci // g] = c.weight[i * co // g:(i + 1) * co // g]
                    nc = nn.Conv2d(ci, co, c.kernel_size, c.stride, c.padding, c.dilation, 1, c.bias is not None).to(c.weight.device)
                    nc.weight.copy_(w)
                    if c.bias is not None:
                        nc.bias.copy_(c.bias)
                    m._modules[k] = nc; n += 1
    return n


def toggle_record(flags, prec: str) -> dict:
    f = set(flags)
    return {"attn_backend": ("sdpa" if "sdpa" in f else "manual") + ("+motion_sdpa" if "sdpa_motion" in f else "+motion_manual"),
            "precision": prec, "bn_folded": "bnfold" in f, "cudnn_benchmark": "cudnn" in f, "channels_last": "chlast" in f,
            "dense_group_conv": "dense_group" in f,
            "compiled": [x for x in ("c_enc", "c_head", "c_rest") if x in f] or False,
            "refine_hook": "hookoff" not in f, "input_staging": "fp16_before_pad" if "fp16build" in f else "fp32",
            "torch": torch.__version__, "venv": "FocusMamba .venv (unmodified; runtime patches only)"}


# ---- E2a ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def fold_bn(root: nn.Module) -> int:
    n = 0
    for m in root.modules():
        if not isinstance(m, nn.Sequential):
            continue
        names = list(m._modules)
        for a, b in zip(names, names[1:]):
            c, bn = m._modules[a], m._modules[b]
            if isinstance(c, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d) and not bn.training:
                s = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
                w = c.weight.double() * s.view(-1, 1, 1, 1)
                bias = (bn.bias.double() - bn.running_mean.double() * s) + (0 if c.bias is None else c.bias.double() * s)
                nc = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, c.dilation, c.groups, True)
                nc.weight.copy_(w.to(c.weight.dtype)); nc.bias.copy_(bias.to(c.weight.dtype))
                m._modules[a] = nc.to(c.weight.device); m._modules[b] = nn.Identity()
                n += 1
    return n


# ---- E2 restorer forward (stage 1 / 2) with optional fp16 build + channels_last ---------------------------------------------
def _make_restore(net, fp16build: bool, chlast: bool):
    from models.shared_temporal import triplets

    def build(padded):
        if fp16build and torch.is_autocast_enabled():
            st = triplets(padded.half())                    # identical values to autocast's per-conv fp32->fp16 cast
        else:
            st = triplets(padded)
        return st.contiguous(memory_format=torch.channels_last) if chlast else st

    def blk_res(blk, st, centre):                           # residual head of one block for one chunk
        x0 = blk.input_block(st)
        x1 = blk.down_0(x0)
        x2 = blk.up_1(blk.down_1(x1))
        x1 = blk.up_0(x1 + x2)
        return blk.output_block(x0 + x1)

    def restore_clip(self, clip, feats):
        assert self.proj is None or feats is None, "injection path not patched"
        n, _, h, w = clip.shape
        ph, pw = -h % 4, -w % 4
        padded = F.pad(clip, (0, pw, 0, ph), mode="reflect") if ph or pw else clip
        st, out = build(padded), []
        for s in range(0, n, self.batch):
            r = blk_res(self.temp2, st[s:s + self.batch], None)
            out.append((padded[s:s + self.batch] - r).clamp(0, 1)[..., :h, :w])   # centre frame in fp32 as in the pinned code
        return torch.cat(out)

    def restore_two_stage(self, clip):
        r1 = self.restore_clip(clip, None)
        if self.block2 is None:
            return r1
        n, _, h, w = r1.shape
        ph, pw = -h % 4, -w % 4
        padded = F.pad(r1, (0, pw, 0, ph), mode="reflect") if ph or pw else r1
        st, res = build(padded), []
        for s in range(0, n, self.batch):
            res.append(blk_res(self.block2, st[s:s + self.batch], None)[..., :h, :w])
        return (r1 - self.stage2_scale * self.gate2.g * torch.cat(res)).clamp(0, 1)

    net.restore_clip = types.MethodType(restore_clip, net)
    net.restore_two_stage = types.MethodType(restore_two_stage, net)


def apply(net, flags, is_c3: bool):
    """Apply a set of flags to an already-built, eval, on-device net (C3 SharedTemporalNet or upstream VideoDepthAnything)."""
    flags = set(flags)
    unknown = flags - set(E_FLAGS) - set(A_FLAGS)
    assert not unknown, unknown
    log = {}
    if "sdpa" in flags:
        patch_sdpa()
    if "sdpa_motion" in flags:
        patch_sdpa_motion()
    if "cudnn" in flags:
        torch.backends.cudnn.benchmark = True
    vda = net.vda.core if is_c3 else net
    if is_c3:
        if "hookoff" in flags:
            hm = vda.head.scratch.refinenet1
            hm._forward_hooks.clear(); hm._st_out = []
            log["hooks_left"] = len(hm._forward_hooks)
        if "bnfold" in flags:
            log["bn_folded"] = fold_bn(net.temp2) + (fold_bn(net.block2) if net.block2 is not None else 0)
        if "dense_group" in flags:
            log["dense_group"] = dense_group(net.temp2) + (dense_group(net.block2) if net.block2 is not None else 0)
        if "chlast" in flags:
            net.temp2.to(memory_format=torch.channels_last)
            if net.block2 is not None:
                net.block2.to(memory_format=torch.channels_last)
        if flags & {"fp16build", "chlast"}:
            _make_restore(net, "fp16build" in flags, "chlast" in flags)
    if flags & set(A_FLAGS):
        import torch._inductor.config as ic
        ic.freezing = True
        for p in net.parameters():
            p.requires_grad_(False)
        if "c_enc" in flags:
            vda.pretrained.get_intermediate_layers = torch.compile(vda.pretrained.get_intermediate_layers, dynamic=False)
        if "c_head" in flags:
            vda.head.forward = torch.compile(vda.head.forward, dynamic=False)
        if "c_rest" in flags and is_c3:
            for blk in [net.temp2, net.block2]:
                if blk is not None:
                    for nm in ("input_block", "down_0", "down_1", "up_0", "up_1", "output_block"):
                        getattr(blk, nm).compile(dynamic=False)
    return log
