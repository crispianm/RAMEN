"""RAMEN-L (ViT-L) guards: sha pin, encoder mix-up refusal, trunk checkpointing equivalence, restorer widths.
Needs a checkout of Video-Depth-Anything: set RAMEN_VDA_REPO (tests skip otherwise). CPU only, tiny inputs."""
from __future__ import annotations

import hashlib
import os
import sys

import pytest
import torch

REPO = os.environ.get("RAMEN_VDA_REPO", "")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, "video_depth_anything")), reason="RAMEN_VDA_REPO not set")


def _vda(**kw):
    from models.vda_net import VDANet
    return VDANet(repo=REPO, init="random", metric=True, **kw)


def test_encoder_dims_vits_and_vitl():
    s, l = _vda(encoder="vits"), _vda(encoder="vitl")
    assert (s.core.pretrained.embed_dim, len(s.core.pretrained.blocks), s.core.head.scratch.output_conv1.in_channels) == (384, 12, 64)
    assert (l.core.pretrained.embed_dim, len(l.core.pretrained.blocks), l.core.head.scratch.output_conv1.in_channels) == (1024, 24, 256)
    assert l.core.intermediate_layer_idx["vitl"] == [4, 11, 17, 23]
    assert [m.temporal_transformer.norm.num_channels for m in l.core.head.motion_modules] == [1024, 1024, 256, 256]
    assert [m.temporal_transformer.norm.num_channels for m in s.core.head.motion_modules] == [192, 384, 64, 64]


def test_vits_state_dict_judged_as_vitl_fails(tmp_path):
    """NEGATIVE CONTROL: a ViT-S checkpoint presented as ViT-L must be refused (strict load)."""
    from models.vda_net import VDANet
    p = tmp_path / "s.pth"
    torch.save(_vda(encoder="vits").core.state_dict(), p)
    with pytest.raises(RuntimeError):
        VDANet(repo=REPO, encoder="vitl", metric=True, init="released", checkpoint_path=str(p))
    VDANet(repo=REPO, encoder="vits", metric=True, init="released", checkpoint_path=str(p))   # positive control: same file as ViT-S loads


def test_wrong_sha_refuses_and_right_sha_loads(tmp_path):
    from models.vda_net import VDANet
    p = tmp_path / "s.pth"
    torch.save(_vda(encoder="vits").core.state_dict(), p)
    good = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="REFUSED"):
        VDANet(repo=REPO, encoder="vits", metric=True, init="released", checkpoint_path=str(p), checkpoint_sha256="0" * 64)
    VDANet(repo=REPO, encoder="vits", metric=True, init="released", checkpoint_path=str(p), checkpoint_sha256=good)


def test_vitl_never_falls_back_to_vits_file(tmp_path):
    from models.vda_net import VDANet
    with pytest.raises(FileNotFoundError):
        VDANet(repo=REPO, encoder="vitl", metric=True, init="released", checkpoint_path=str(tmp_path / "missing.pth"))


def test_env_var_weights_path(tmp_path, monkeypatch):
    from models.vda_net import VDANet
    torch.save(_vda(encoder="vits").core.state_dict(), tmp_path / "w.pth")
    monkeypatch.setenv("RAMEN_WEIGHTS", str(tmp_path))
    VDANet(repo=REPO, encoder="vits", metric=True, init="released", checkpoint_path="${RAMEN_WEIGHTS}/w.pth")


def test_trunk_checkpointing_is_numerically_identical():
    torch.manual_seed(0)
    a = _vda(encoder="vits")
    b = _vda(encoder="vits", encoder_grad_ckpt=True)
    b.load_state_dict(a.state_dict())
    for m in (a.core.head.motion_modules, b.core.head.motion_modules):   # zero-init temporal: make the head non-trivial
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.02)
    b.load_state_dict({**b.state_dict(), **{k: v for k, v in a.state_dict().items() if "motion_modules" in k}})
    x = torch.rand(1, 3, 32, 28, 28)
    out = []
    for net in (a, b):
        xx = x.clone().requires_grad_(True)
        d = net(xx)
        d.square().mean().backward()
        out.append((d.detach(), xx.grad.clone(), next(p.grad.clone() for p in net.core.head.parameters() if p.grad is not None)))
    for u, v in zip(*out):
        torch.testing.assert_close(u, v, rtol=1e-4, atol=1e-6)
    assert getattr(b.core.pretrained.blocks[0], "_ramen_ckpt", False) and not getattr(a.core.pretrained.blocks[0], "_ramen_ckpt", False)


def test_checkpointing_inactive_under_no_grad():
    b = _vda(encoder="vits", encoder_grad_ckpt=True).eval()
    with torch.no_grad():
        b(torch.rand(1, 3, 32, 28, 28))


def test_restorer_widths():
    from models.shared_temporal import WIDTHS, temp2_block
    n = {w: sum(p.numel() for p in temp2_block(w).parameters()) for w in ("full", "wide2x", "wide3x", "wide4x")}
    assert n["full"] == 321_618 and n["wide2x"] == 1_248_882           # the values quoted in shared_temporal.py
    assert n["wide2x"] < n["wide3x"] < n["wide4x"]
