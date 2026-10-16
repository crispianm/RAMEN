"""DN-01 WS2 arm S (K49 static-clip regime): ``data.static_prob`` of the labelled training clips become ONE still frame +
its GT repeated T times, every frame degraded INDEPENDENTLY (a per-frame noise multiplier U[0.5, 1.5], and for lowlight an
EV jitter U[-0.15, +0.15] stops; the exposed target follows each frame's EV). The static coin is drawn from its OWN RNG,
so every non-static clip is the control's clip (header dn01_ws2.yaml, section 2)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from data.elvis_noise import ELVIS_PHI_PATHS, lowlight_reference

needs_bank = pytest.mark.skipif(not any(os.path.isfile(p) for p in ELVIS_PHI_PATHS), reason="needs the ELVIS bank")
LL = {"ev": [-3.0, -1.0], "kappa": 0.5, "target": "exposed"}
T, S = 16, 32


def _ds(mode, s_max=1.0, split="train"):
    """The real mixin over a stand-in clean parent that returns the labelled-clip keys of the VDA datasets."""
    from data.vda_semantics_dataset import _VDADegradedMixin

    class Base:
        degrade, seed, degrade_severity, degrade_mode = True, 0, None, mode
        severity_max = torch.full((1,), float(s_max))

        def __getitem__(self, idx):
            g = torch.Generator().manual_seed(idx)
            return {"frames": torch.rand(3, T, S, S, generator=g) * 0.6 + 0.2,
                    "depth": torch.rand(1, T, S, S, generator=g) * 50, "mask": torch.ones(1, T, S, S),
                    "sky": torch.zeros(1, T, S, S), "video_id": "v", "start_frame": idx}

    class DS(_VDADegradedMixin, Base):
        seeded_noise = True
    ds = DS()
    ds.split = split
    return ds


def _get(ds, i):
    torch.manual_seed(i)
    return ds[i]


def _parent(ds, i):
    """The clean parent's item (what the mixin degrades)."""
    from data.vda_semantics_dataset import _VDADegradedMixin
    return super(_VDADegradedMixin, ds).__getitem__(i)


def test_the_control_data_path_is_untouched_when_the_regime_is_off():
    """static_prob 0 (every non-S arm and the reused control): the item's KEY SET is the pre-WS2 one (no `static`)."""
    ds = _ds("awgn_blur")
    got = _get(ds, 0)
    assert "static" not in got and set(got) == {"frames", "depth", "mask", "sky", "video_id", "start_frame",
                                                "degraded_frames", "degrade_params"}


def test_every_item_carries_the_static_key_when_the_regime_is_on():
    """The collate key-set contract: with static_prob > 0 every degraded item (static or not; val too) has `static`."""
    for split in ("train", "val"):
        ds = _ds("awgn_blur", split=split)
        ds.static_prob = 0.25
        assert all("static" in _get(ds, i) for i in range(6))


def test_non_static_clips_are_the_controls_clips():
    """The coin has its own RNG: at static_prob 0.5 every item the coin left non-static equals the static_prob-0 item."""
    ctrl, ds = _ds("awgn_blur"), _ds("awgn_blur")
    ds.static_prob = 0.5
    kinds = set()
    for i in range(24):
        a, b = _get(ctrl, i), _get(ds, i)
        kinds.add(int(b["static"]))
        if not float(b["static"]):
            for k in ("frames", "degraded_frames", "degrade_params", "depth", "mask", "sky"):
                assert torch.equal(a[k], b[k]), k
    assert kinds == {0, 1}


def test_static_clip_repeats_one_frame_and_its_gt_and_degrades_each_frame_independently():
    ds = _ds("awgn_blur")
    ds.static_prob, ds.degrade_severity = 1.0, 0.5
    got, parent = _get(ds, 3), _parent(ds, 3)
    assert float(got["static"]) == 1.0
    for k in ("frames", "depth", "mask", "sky"):                             # frame T // 2 and its GT, x T
        want = parent[k][:, T // 2:T // 2 + 1].expand(-1, T, -1, -1)
        assert torch.equal(got[k], want), k
    x = got["degraded_frames"]
    assert x.shape == got["frames"].shape
    assert all(not torch.equal(x[:, 0], x[:, t]) for t in range(1, T))      # a fresh noise draw per frame


def test_static_clips_carry_a_per_frame_noise_multiplier():
    """awgn (sigma = s * 50/255): the per-frame residual std varies as m_t ~ U[0.5, 1.5] around the clip's own sigma."""
    ds = _ds("awgn")
    ds.static_prob, ds.degrade_severity = 1.0, 0.4
    got = _get(ds, 5)
    res = (got["degraded_frames"] - got["frames"]).clamp(-1, 1)
    std = res.reshape(3, T, -1).std((0, 2)).numpy()
    base = 0.4 * 50 / 255
    assert (std > 0.4 * base).all() and (std < 1.6 * base).all()             # inside m in [0.5, 1.5] (+ sampling slack)
    assert std.max() / std.min() > 1.3                                       # and it really varies (T = 16 draws)


def test_static_fraction_follows_static_prob_and_val_is_never_static():
    ds = _ds("awgn_blur")
    ds.static_prob = 0.25
    frac = np.mean([float(_get(ds, i)["static"]) for i in range(200)])
    assert 0.15 < frac < 0.35
    val = _ds("awgn_blur", split="val")
    val.static_prob = 1.0
    assert float(val[0]["static"]) == 0.0


@needs_bank
def test_lowlight_static_clip_ev_jitter_and_the_exposed_target_follows_each_frame():
    ds = _ds("lowlight")
    ds.lowlight = dict(LL)
    ds.static_prob, ds.degrade_severity = 1.0, 0.5
    got = _get(ds, 7)
    clean = _parent(ds, 7)["frames"]
    tgt_mean = got["frames"].mean((0, 2, 3)).numpy()                        # (T,) the per-frame target brightness
    inp_mean = got["degraded_frames"].mean((0, 2, 3)).numpy()
    flat = got["frames"].permute(1, 0, 2, 3).reshape(T, -1).numpy()
    assert min(np.corrcoef(flat[0], flat[t])[0, 1] for t in range(1, T)) > 0.999    # ONE still: same pixels, up to gain
    assert tgt_mean.std() / tgt_mean.mean() > 0.02                           # EV jitter (+-0.15 stops ~ 6% gain std)
    assert np.corrcoef(tgt_mean, inp_mean)[0, 1] > 0.9                       # the input follows the same per-frame EV
    ev = float(np.log2(got["degrade_params"][2]))                            # the clip-level EV (jitter is around it)
    mid = clean[:, T // 2:T // 2 + 1].permute(1, 0, 2, 3).numpy()
    lo, hi = (lowlight_reference(mid, ev + d).mean() for d in (-0.15, 0.15))
    assert lo - 1e-3 <= tgt_mean.min() and tgt_mean.max() <= hi + 1e-3       # the target stays inside EV +- 0.15 stops


def test_static_regime_config_is_checked():
    from train import check_static_regime
    check_static_regime({"data": {"degrade_mode": ["awgn_blur", "lowlight"], "static_prob": 0.25}})
    check_static_regime({"data": {"degrade_mode": "awgn_blur"}})             # absent = off
    for bad in ({"static_prob": 1.5}, {"static_prob": -0.1}, {"static_prob": 0.25, "degrade_mode": "full"},
                {"static_prob": 0.25, "degrade_mode": ["awgn_blur", "elvis"]},
                {"static_prob": 0.25, "degrade_mode": "awgn_blur", "static_noise_mult": [1.5, 0.5]}):
        with pytest.raises(ValueError):
            check_static_regime({"data": {"degrade_mode": "awgn_blur", **bad}})


def test_rg_loss_weight_is_one_on_static_clips_for_every_guide():
    import train
    g = torch.Generator().manual_seed(0)
    pred = torch.rand(2, 1, 6, 12, 12, generator=g) * 40 + 2
    guide = torch.rand(2, 3, 6, 12, 12, generator=g)                          # large |dG| everywhere
    plain, guided = train.rg_loss(pred, None, 0.03), train.rg_loss(pred, guide, 0.03)
    assert guided < 0.5 * plain                                               # the guide really down-weights
    both = train.rg_loss(pred, guide, 0.03, static=torch.ones(2))
    assert torch.allclose(both, plain)                                        # all-static: weight 1 whatever the guide
    none = train.rg_loss(pred, guide, 0.03, static=torch.zeros(2))
    assert torch.allclose(none, guided)                                       # no static clip: unchanged
    mixed = train.rg_loss(pred, guide, 0.03, static=torch.tensor([1.0, 0.0]))
    assert guided < mixed < plain
