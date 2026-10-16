"""Protocol v3 WP17: the design-(c) coupling (headc 4f1794d) ported onto the a1h base: step-0 depth BIT-IDENTICAL to the a1h base, the keys resolve, and the base arms are untouched. CPU; needs the released
Metric-VDA-S weights + the VDA repo + the released Pocket temp2 on this host (skipped otherwise)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train  # noqa: E402
from models import build_model, trainable_modules  # noqa: E402

LOCAL = {"vda_repo": "/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything", "device": "cpu",
         "vda_checkpoint_path": "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth",
         "da_v2_checkpoint_path": "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth_anything_v2_vits.pth", "st_temp2_init": None}
need = pytest.mark.skipif(not Path(LOCAL["vda_checkpoint_path"]).is_file() or not Path(LOCAL["vda_repo"]).is_dir(), reason="Metric-VDA-S weights / VDA repo not on this host")


def _cfg(name):
    return train.load_config(ROOT / "configs" / "experiments" / f"{name}.yaml")


def _net(name):
    c = _cfg(name)
    c["model"].update(LOCAL)
    return build_model(c).eval(), c


def test_the_b_fixture_differs_from_a1h_only_in_the_coupling_keys():
    a, b = _cfg("dn01_mf_a1h_n45"), _cfg("dn01_v3_headc_b_a1h")
    flat = lambda d, p="": {k2: v2 for k, v in d.items() for k2, v2 in (flat(v, f"{p}{k}.").items() if isinstance(v, dict) else {f"{p}{k}": v}.items())}
    fa, fb = flat(a), flat(b)
    diff = {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}
    assert diff == {"model.st_inject", "model.st_depth_input", "model.st_stage2_width", "training.gate_lr_scale", "training.checkpoint_dir"}, sorted(diff)
    assert fb["model.st_inject"] is False and fb["model.st_depth_input"] == "restored" and fb["model.st_stage2_width"] == "wide2x" and fb["training.gate_lr_scale"] == 10


@need
def test_step0_depth_is_bit_identical_to_the_a1h_base():
    base, _ = _net("dn01_mf_a1h_n45")
    b, c = _net("dn01_v3_headc_b_a1h")
    assert base.depth_input == "raw" and b.depth_input == "restored" and b.block2 is not None and float(b.gate.g) == 0.0 and float(b.gate2.g) == 0.0
    # the SAME depth weights (both released Metric-VDA-S, DPT head + motion modules untouched at step 0) and the same restorer stage 1
    for (n1, p1), (n2, p2) in zip(base.vda.state_dict().items(), b.vda.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    x = torch.rand(1, 3, 8, 56, 56)
    with torch.no_grad():
        d0, _ = base.depth_and_feats(x)
        d1, _ = b.depth_and_feats(x)
        r0 = b.restore_batch(x)
        d_plain = base.vda(x)
    assert torch.equal(d0, d1), float((d0 - d1).abs().max())                         # bit for bit
    assert torch.equal(d0, d_plain) and d0.shape == (1, 1, 8, 56, 56)                  # == the plain Metric-VDA-S forward (A never reads R either)
    assert float((r0 - x).abs().max()) > 0                                             # the restorer does something (it is just gated out of the depth path at step 0)


@need
def test_the_gate_sits_in_its_own_lr_group_and_engagement_scales_are_overrides_not_parameters():
    b, c = _net("dn01_v3_headc_b_a1h")
    opt = train.build_optimizer(b, c, verbose=False)
    lrs = sorted({g["lr"] for g in opt.param_groups})
    lr = c["training"]["learning_rate"]
    assert lr in lrs and lr * 10 in lrs                                               # gate_lr_scale 10 -> its own group (also with a1h's head_lr_mult set: that branch is taken first, the port carves the gates out)
    gate_ids = {id(p) for m in (b.gate, b.gate2) for p in m.parameters()}
    hi = [g for g in opt.param_groups if g["lr"] == lr * 10]
    assert len(hi) == 1 and {id(p) for p in hi[0]["params"]} == gate_ids and len(gate_ids) == 2                  # exactly the two scalar gates
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == sum(p.numel() for p in b.parameters() if p.requires_grad)      # every trainable tensor in exactly one group
    assert not any("coupling_scale" in n or "stage2_scale" in n for n, _ in b.named_parameters())
    x = torch.rand(1, 3, 8, 56, 56)
    with torch.no_grad():
        b.gate.g.fill_(0.5)
        full = b.depth_and_feats(x)[0]
        b.coupling_scale = 0.0
        raw = b.depth_and_feats(x)[0]
    base, _ = _net("dn01_mf_a1h_n45")
    with torch.no_grad():
        assert torch.equal(raw, base.depth_and_feats(x)[0])                            # coupling_scale 0 == depth(R := raw) == the a1h base exactly
    assert float((full - raw).abs().max()) > 0                                         # and a non-zero gate does change depth (the coupling is real)


@need
def test_the_a1h_base_arm_is_unchanged_by_the_port():
    base, c = _net("dn01_mf_a1h_n45")
    assert base.depth_input == "raw" and base.gate is None and base.block2 is None
    assert "gate" not in dict(trainable_modules(base)) and "gate_lr_scale" not in c["training"]


@need
def test_gate_group_is_lr_x10_and_every_other_group_is_unchanged():
    """Auditor WP25 pre-audit item 4: with gate_lr_scale 10 the ONLY change to the optimizer vs the same net at gate_lr_scale 1 (control) is that the two gates move out of the base group into a group at lr x 10; every other
    group (the head_lr_mult group and the rest) keeps its lr and its tensors. Also: both optimizers cover every trainable tensor exactly once."""
    b, c = _net("dn01_v3_headc_b_a1h")
    lr = c["training"]["learning_rate"]
    c1 = copy.deepcopy(c)
    c1["training"]["gate_lr_scale"] = 1.0
    ctrl = train.build_optimizer(b, c1, verbose=False)
    x10 = train.build_optimizer(b, c, verbose=False)
    ids = lambda g: [id(p) for p in g["params"]]
    gates = {id(p) for m in (b.gate, b.gate2) for p in m.parameters()}
    assert len(gates) == 2 and len(ctrl.param_groups) == 2 and len(x10.param_groups) == 3                  # control: rest + head_lr_mult group; x10: + the gate group
    rest_c, head_c = ctrl.param_groups
    rest_x, head_x, gate_x = x10.param_groups
    assert head_c["lr"] == head_x["lr"] == lr * c["training"]["head_lr_mult"] and ids(head_c) == ids(head_x)       # the head group: same lr, same tensors, same order
    assert rest_c["lr"] == rest_x["lr"] == lr and set(ids(rest_c)) == set(ids(rest_x)) | gates and not set(ids(rest_x)) & gates   # the rest: same lr; it only lost the gates
    assert gate_x["lr"] == lr * 10 and set(ids(gate_x)) == gates
    all_x = [i for g in x10.param_groups for i in ids(g)]
    assert len(all_x) == len(set(all_x)) == len([p for p in b.parameters() if p.requires_grad])
    for g in x10.param_groups:                                                                                # the other hyper-parameters of the new group follow the optimizer defaults
        assert g["weight_decay"] == ctrl.param_groups[0]["weight_decay"] and g["betas"] == ctrl.param_groups[0]["betas"]
