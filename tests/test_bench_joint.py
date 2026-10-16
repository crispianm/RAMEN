"""bench_joint.py: its reimplemented pieces equal the originals they claim to be."""
from pathlib import Path

import numpy as np
import pytest
import torch

import bench_joint as bj

needs_vda = pytest.mark.skipif(not Path(bj.PATHS["vda_repo"]).is_dir(), reason="VDA repo not on this host")
FDVD = Path(bj.PATHS["fastdvdnet_repo"])
needs_fdvd = pytest.mark.skipif(not (FDVD / "model_clipped_noise.pth").is_file(), reason="FastDVDnet not on this host")


class _FakeVDA:
    """A stand-in for VideoDepthAnything whose depth scale changes per window, so the
    scale/shift stitching is exercised."""
    def __init__(self, metric):
        self.metric = metric

    def forward(self, x):                                   # (1,T,3,H,W) normalised
        return (x.mean(2) + 3.0) * (1.5 + x[:, :, 0].mean())


@needs_vda
@pytest.mark.parametrize("metric", [False, True])
def test_vda_windows_equals_official_infer_video_depth(metric):
    bj._vda_path()
    from video_depth_anything.video_depth import VideoDepthAnything
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 255, size=(60, 28, 28, 3)).astype(np.float32)   # 3 windows, padded tail
    fake = _FakeVDA(metric)
    ref, _ = VideoDepthAnything.infer_video_depth(fake, frames, 1, input_size=28, device="cpu", fp32=True)
    clip = torch.from_numpy(frames / 255.0).permute(0, 3, 1, 2).float()
    restored_in = clip + 1.0                                # restored output = input + 1, to track frames
    got, restored = bj.vda_windows(lambda w: (fake.forward(bj._norm(w)[None])[0], w + 1.0), clip, (28, 28), metric)
    assert got.shape == ref.shape == (60, 28, 28)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(restored, restored_in)       # every frame once, in order, padding dropped


@needs_fdvd
def test_fastdvdnet_batched_equals_official_and_denoises():
    net = bj._fastdvdnet("cpu")["net"]
    official = bj._load_file("fastdvdnet_algo", FDVD / "fastdvdnet.py").denoise_seq_fastdvdnet
    g = torch.Generator().manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 30), torch.linspace(0, 1, 34), indexing="ij")
    clean = torch.stack([torch.stack([yy, xx, (yy + xx) / 2]) * (0.6 + 0.05 * t) for t in range(7)])
    noisy = (clean + torch.randn(clean.shape, generator=g) * 25 / 255).clamp(0, 1)
    with torch.no_grad():
        ref = official(noisy, torch.tensor([25 / 255]), 5, net)
        got = bj.fastdvdnet_denoise(net, noisy, 25.0, batch=3)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    assert (got - clean).pow(2).mean() < 0.5 * (noisy - clean).pow(2).mean()


def test_fastdvdnet_short_clips_clamp_and_long_clips_are_unchanged():
    """Single images (KITTI-C / NYUv2 held-out sets): the reflected neighbour index went negative for n < 3 (a CUDA
    device assert); it is now clamped like PocketFrontEnd's. For n >= 3 the indices are exactly the old ones."""
    for n in range(3, 10):
        idx = torch.arange(n)[:, None] + torch.arange(-2, 3)[None]
        idx = torch.where(idx < 0, -idx, idx)
        old = torch.where(idx > n - 1, 2 * (n - 1) - idx, idx)
        assert torch.equal(old, old.clamp(0, n - 1)) and old.min() >= 0 and old.max() <= n - 1
    net = bj._fastdvdnet("cpu")["net"]
    x = torch.rand(1, 3, 20, 24, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        one = bj.fastdvdnet_denoise(net, x, 25.0)
        two = bj.fastdvdnet_denoise(net, torch.cat([x, x]), 25.0)
        same = bj.fastdvdnet_denoise(net, x.expand(5, -1, -1, -1).contiguous(), 25.0)[2:3]
    assert one.shape == x.shape and two.shape == (2, *x.shape[1:])
    torch.testing.assert_close(one, same)                   # n = 1: all five neighbours are the frame itself


@needs_vda
def test_ours_loader_equals_training_model_forward(tmp_path):
    from models import build_model, trainable_modules
    cfg = {"model": {"arch": "vda", "vda_init": "random", "restoration_head": True}}
    torch.manual_seed(0)
    net = build_model(cfg).eval()
    with torch.no_grad():                                   # trained-looking head weights, not the init
        for p in torch.nn.ModuleDict(trainable_modules(net)).parameters():
            p.add_(0.01 * torch.randn_like(p))
    ck = tmp_path / "step1.pt"
    torch.save({"cfg": cfg, "trainable_ema": torch.nn.ModuleDict(trainable_modules(net)).state_dict()}, ck)
    torch.manual_seed(0)                                    # same random frozen backbone
    p = bj._ours("cpu", str(ck))
    win = torch.rand(32, 3, 28, 28)
    with torch.no_grad():
        d_ref, r_ref = net(win.permute(1, 0, 2, 3)[None], return_recon=True)
        d, r = p["fwd"](win)
    assert p["metric"] is False and p["window"] == 32
    torch.testing.assert_close(d, d_ref[0, 0])
    torch.testing.assert_close(r, r_ref[0].permute(1, 0, 2, 3).clamp(0, 1))


def test_score_clip_perfect_prediction_and_degrade_determinism():
    g = torch.Generator().manual_seed(1)
    gt = (1.0 + 40.0 * torch.rand(5, 20, 24, generator=g)).double().numpy()
    gt[:, :2] = -1                                          # invalid rows, as VDA's get_gt marks them
    clean = torch.rand(5, 3, 14, 14, generator=g)
    x = bj.degrade(clean, 1.5, 25.0, seed=7)
    assert torch.equal(x, bj.degrade(clean, 1.5, 25.0, seed=7)) and torch.equal(bj.degrade(clean, 0, 0, 7), clean)
    rel = bj.score_clip(3.0 / np.abs(gt) + 0.2, False, gt, clean, x, clean, 80.0, 0.1)   # affine in disparity
    assert rel["abs_rel_aligned"] < 1e-6 and rel["delta1_aligned"] == 1.0 and rel["joint_p95"] < 1e-6
    assert rel["psnr"] > 100 and rel["psnr_input"] < 40 and "abs_rel_unaligned" not in rel
    met = bj.score_clip(np.abs(gt), True, gt, clean, x, clean, 80.0, 0.1)
    assert met["abs_rel_unaligned"] < 1e-6 and met["abs_rel_aligned"] < 1e-6


def test_estimate_sigma_recovers_awgn():
    g = torch.Generator().manual_seed(2)
    yy, xx = torch.meshgrid(torch.linspace(0.3, 0.7, 64), torch.linspace(0.3, 0.7, 64), indexing="ij")
    clean = torch.stack([yy, xx, (yy + xx) / 2])[None].repeat(4, 1, 1, 1)
    noisy = clean + torch.randn(clean.shape, generator=g) * 25 / 255
    assert bj.estimate_sigma(noisy) == pytest.approx(25, rel=0.05)
    assert bj.estimate_sigma(clean) < 1


def test_restorer_chain_order(monkeypatch):
    calls = []
    fake = lambda tag: (lambda dev: dict(fn=lambda x, s, b: (calls.append(tag), x + 1)[1], weights={tag: tag}))
    monkeypatch.setitem(bj.DENOISERS, "r1", fake("r1"))
    monkeypatch.setitem(bj.DENOISERS, "r2", fake("r2"))
    monkeypatch.setitem(bj.DEPTH_MODELS, "d", lambda dev: dict(fwd=None, metric=False, window=1, weights={}))
    p = bj.build_pipeline("r1+r2+d", "cpu")
    assert torch.equal(p["denoise"](torch.zeros(1), 25.0, 3.0), torch.full((1,), 2.0)) and calls == ["r1", "r2"]
    assert list(p["weights"]) == ["r1", "r2"]


def test_post_restorer_runs_on_restored_output_only(monkeypatch):
    seen = []
    fake = lambda tag: (lambda dev: dict(fn=lambda x, s, b: x + 1, weights={tag: tag}))
    monkeypatch.setitem(bj.DENOISERS, "r1", fake("r1"))
    monkeypatch.setitem(bj.DENOISERS, "r2", fake("r2"))
    monkeypatch.setitem(bj.DEPTH_MODELS, "d", lambda dev: dict(
        fwd=lambda x: (seen.append(x.clone()), x.mean(1))[1:] + (None,), metric=False, window=1, weights={}))
    assert bj.split_spec("r1+d+r2") == (["r1"], "d", ["r2"])
    assert bj.split_spec("ours:a/runs/x/step20000.pt+r2") == ([], "ours:a/runs/x/step20000.pt", ["r2"])
    p = bj.build_pipeline("r1+d+r2", "cpu")
    x = torch.zeros(2, 3, 4, 4)
    depth, restored = bj.run_pipeline(p, x, 25.0, 3.0, (4, 4))
    assert torch.equal(seen[0], x + 1) and torch.equal(restored, x + 2)       # depth saw r1 only; r2 after it
    assert list(p["weights"]) == ["r1", "r2"]
    assert bj.short_name("ours:c/runs/dn01_x/step20000.pt+r2") == "dn01_x@20000+r2"
    assert bj.latency_of({"ours:c/runs/dn01_h8_x/step1.pt": 6.4}, "ours:c/runs/dn01_h8_x/step2.pt+r2") is None


def test_disk_cells_are_held_out_and_second_moment_matched():
    assert bj.parse_cell("b3s25") == (3.0, 25.0) and not isinstance(bj.parse_cell("b3s25")[0], bj.Disk)
    blur, sigma = bj.parse_cell("d6s25")
    assert isinstance(blur, bj.Disk) and (blur, sigma) == (6.0, 25.0)
    assert not any(c.startswith("d") for c in bj.CELLS)                       # off by default
    k = bj.disk_psf(6.0)
    xx = np.arange(k.shape[0]) - k.shape[0] // 2
    assert k.sum() == pytest.approx(1.0) and (k * xx[None] ** 2).sum() ** 0.5 == pytest.approx(3.0, rel=0.05)
    g = torch.Generator().manual_seed(4)
    clean = torch.rand(2, 3, 40, 44, generator=g)
    d = bj.degrade(clean, blur, 0.0, seed=0)
    assert torch.equal(d, bj.degrade(clean, bj.Disk(6), 0.0, seed=0)) and not torch.equal(d, bj.degrade(clean, 3.0, 0, 0))


@pytest.mark.skipif(not (Path(bj.PATHS["restormer_repo"]) / "Defocus_Deblurring/pretrained_models/"
                         "single_image_defocus_deblurring.pth").is_file(), reason="Restormer not on this host")
def test_restormer_skips_without_blur_and_restores_with_it():
    fn = bj._restormer("cpu")["fn"]
    x = torch.rand(2, 3, 20, 22)
    assert fn(x, 25.0, 0.0) is x
    with torch.no_grad():
        y = fn(x, 25.0, 3.0)
    assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1 and not torch.equal(y, x)


def test_wiener_oracle_inverts_the_benchmark_blur():
    g = torch.Generator().manual_seed(3)
    clean = torch.nn.functional.avg_pool2d(torch.rand(2, 3, 96, 104, generator=g), 3, 1, 1)   # textured
    fn = bj._wiener_oracle("cpu")["fn"]
    assert fn(clean, 25.0, 0.0) is clean
    for blur in (1.5, 3.0, bj.Disk(6.0)):                  # the disk cells' oracle uses the pillbox PSF
        blurred = bj.degrade(clean, blur, 0.0, seed=0)
        restored = fn(blurred, 0.0, blur)
        assert bj.psnr(restored, clean).mean() > bj.psnr(blurred, clean).mean() + 0.5   # K is tuned for noise


def test_plot_and_table_read_results_and_latency(tmp_path, capsys):
    import json
    ours = "ours:/x/runs/dn01_ft_h8_blur_cmode/step20000.pt"
    for m, ms, rel, db in [(bj.BASELINE, 9.0, 0.096, 21.5), ("fastdvdnet+restormer+vda_s", 60.0, 0.090, 23.0),
                           (ours, 6.2, 0.085, 24.0)]:
        for ds in ("kitti", "bonn"):
            vm = {"abs_rel_aligned": rel, "delta1_aligned": 0.9, "psnr": db, "ssim": 0.8, "eps_r": 0.05,
                  "joint_p95": 0.2}
            (tmp_path / f"{bj._slug(m)}__{ds}__b3s25__dev.json").write_text(json.dumps(
                {"model": m, "dataset": ds, "cell": "b3s25", "split": "dev", "video_macro": vm}))
        (tmp_path / f"latency__{bj._slug(m)}.json").write_text(json.dumps({"model": m, "per_frame_ms_p50": ms}))
    ns = type("A", (), {"out": str(tmp_path), "cell": "b3s25", "split": "dev"})
    bj.cmd_plot(ns)
    assert (tmp_path / "latency_vs_quality_b3s25_dev.png").stat().st_size > 10_000
    assert len((tmp_path / "latency_vs_quality_b3s25_dev.csv").read_text().splitlines()) == 7
    bj.cmd_table(ns)
    assert "| 6.2 | Y Y n |" in capsys.readouterr().out          # ours row: latency + beats flags
    assert bj.short_name(ours) == "dn01_ft_h8_blur_cmode@20000" and bj.short_name(bj.BASELINE) == bj.BASELINE
    lat = {"ours:/l/dn01_fp_h8_blur/step6000.pt": 6.3, "ours:/l/dn01_a1_joint/step20000.pt": 6.8, "vda_s": 6.0}
    assert bj.latency_of(lat, ours) == 6.3                                   # same head (h8), other run
    assert bj.latency_of(lat, "ours:/p/dn01_c1_lse/step12000.pt") == 6.8     # untagged -> h32
    assert bj.latency_of(lat, "ours:/p/dn01_fp_h16_blur/step8000.pt") is None and bj.latency_of(lat, "x") is None
    lat["fastdvdnet+vda_s"] = 15.0
    assert bj.latency_of(lat, "fastdvdnet+metric_vda_s") == 15.0           # same backbone: reuses the relative chain
    assert bj.latency_of(lat, "metric_vda_s") == 6.0
    lat["metric_vda_s"] = 6.1
    assert bj.latency_of(lat, "metric_vda_s") == 6.1                       # its own timing wins when it exists


def test_latency_times_the_depth_post_filter(monkeypatch):
    """cmd_latency's stage list must contain the spec's depth post-filter (agf/ema/...): it used to be left out, so
    ``latency --model <depth>+agf0.2t0.06`` was filed under the agf name with the bare model's time."""
    monkeypatch.setitem(bj.DEPTH_MODELS, "fakedepth", lambda device: {"fwd": lambda y: (y.mean(1), None), "window": 32, "metric": True, "weights": {}})
    assert bj.depth_post_stages(bj.build_pipeline("fakedepth", "cpu"), 16, "cpu") == []       # no token: nothing extra, bare model unchanged
    p = bj.build_pipeline("fakedepth+agf0.2t0.06", "cpu")
    calls = []
    real = p["depth_post"]
    p["depth_post"] = lambda d, g=None: (calls.append((d.shape, d.dtype, type(d))), real(d, g))[1]
    stages = bj.depth_post_stages(p, 16, "cpu")
    assert [n for n, _ in stages] == ["agf0.2t0.06"]
    x = torch.rand(bj.INFER_LEN, 3, 16, 16)
    assert stages[0][1](x, 25.0, 3.0) is x                                                     # passes the frames through
    assert calls == [((bj.INFER_LEN, 16, 16), np.dtype("float32"), np.ndarray)]               # the numpy filter ran on a full window
    p2 = bj.build_pipeline("fakedepth+ema0.7+agf0.2t0.06", "cpu")
    assert [n for n, _ in bj.depth_post_stages(p2, 16, "cpu")] == ["ema0.7+agf0.2t0.06"]
    import inspect
    assert "depth_post_stages(p," in inspect.getsource(bj.cmd_latency)                        # wired into the timed stage list (cmd_latency itself needs a 4090)
    assert '"depth_post_probe"' in inspect.getsource(bj.cmd_latency)                          # the probe resolution / content is recorded in the latency JSON


# --- ultrareview: golden values for score_clip & co. on an IMPERFECT prediction (the perfect-prediction test above cannot
# see a wrong formula: every error term is 0). Pinned from the scorer at commit 328ea76, whose AbsRel / delta1 / rmse were
# replayed against VDA's benchmark/eval/metric.py (see test_depth_errors_equal_vda_metric_py below, which keeps that oracle).
def _golden_inputs():
    rng = np.random.default_rng(20261002)                 # numpy's PCG64 stream is stable across versions
    N, H, W = 8, 20, 24
    gt = rng.uniform(1.0, 70.0, (N, H, W))
    gt[:, :2] = -1                                        # invalid rows, as VDA's get_gt marks them
    gt[3] = -1                                            # one fully invalid frame
    gt[:, 5:9, 5:9] = 95.0                                # beyond the 80 m KITTI cap (inside the 10 m Bonn one after /8)
    depth = gt.clip(1.0, None) * rng.uniform(0.75, 1.3, gt.shape)
    depth[1, 10:11] = 0.5                                 # a near-zero stripe
    depth[6, 12:13] = 150.0                               # a far push
    clean = torch.from_numpy(rng.uniform(0, 1, (N, 3, 14, 14)))
    restored = (clean + torch.from_numpy(rng.normal(0, 0.05, clean.shape))).clamp(0, 1).float()
    degraded = (clean + torch.from_numpy(rng.normal(0, 0.2, clean.shape))).clamp(0, 1).float()
    return depth, gt, restored, degraded, clean.float()


_GOLDEN = {
    "KITTI": dict(abs_rel_aligned=0.5381338228561778, delta1_aligned=0.1586538553237915, rmse_aligned=23.83947384370349, abs_rel_unaligned=0.175254913392112, delta1_unaligned=0.817651093006134, rmse_unaligned=8.178907156669405, eps_d=0.8358500583786591, omega_frac=0.40625, joint_p95=1.0505230179312839, lag_k=0.0, lag_absrel0=0.18886386727413396, fit_scale=0.2569687002053414, fit_shift=0.04183929310778992, eps_r=0.05594459921121597, far_frac80=0.046614583333333334),
    "BONN": dict(abs_rel_aligned=0.5381338289751237, delta1_aligned=0.1586538553237915, rmse_aligned=2.979934216981193, abs_rel_unaligned=0.175254913392112, delta1_unaligned=0.817651093006134, rmse_unaligned=1.0223633945836756, eps_d=0.8358500402697557, omega_frac=0.40625, joint_p95=1.050523026587122, lag_k=0.0, lag_absrel0=0.18886387855507708, fit_scale=0.25696868839447506, fit_shift=0.3347143438566356, eps_r=0.05594459921121597, far_frac80=0.0),
    "REL": dict(abs_rel_aligned=0.5381338228561778, delta1_aligned=0.1586538553237915, rmse_aligned=23.83947384370349, eps_d=0.8358500583786591, omega_frac=0.40625, joint_p95=1.0505230179312839, lag_k=0.0, lag_absrel0=0.18886386727413396, fit_scale=0.2569687002053414, fit_shift=0.04183929310778992, eps_r=0.05594459921121597),
}


@pytest.mark.parametrize("name", ["KITTI", "BONN", "REL"])
def test_score_clip_imperfect_prediction_matches_pinned_golden_values(name):
    depth, gt, r, d, c = _golden_inputs()
    args = {"KITTI": (depth, True, gt, r, d, c, 80.0, 0.1), "BONN": (depth / 8, True, gt / 8, r, d, c, 10.0, 0.1),
            "REL": (1.0 / depth.clip(1e-3), False, gt, r, d, c, 80.0, 0.1)}[name]
    out = bj.score_clip(*args)
    for k, v in _GOLDEN[name].items():
        assert out[k] == pytest.approx(v, rel=1e-7, abs=1e-12), (name, k)
    assert ("abs_rel_unaligned" in out) == (name != "REL")                  # relative models have no unaligned fields


def test_flicker_static_and_depth_ema_golden_values():
    depth, gt, *_ = _golden_inputs()
    st = bj.flicker_static(depth, np.repeat(gt[:1].clip(0.5), 8, 0), 80.0, 0.1, t0=2)
    assert st["flicker"] == pytest.approx(0.8631774592196982, rel=1e-9)
    assert st["abs_rel_static"] == pytest.approx(1.6246655407898867, rel=1e-9)
    e = bj.depth_ema(depth, 0.3)
    assert e[0] == pytest.approx(depth[0], rel=1e-12) and float(e.sum()) == pytest.approx(121104.13078340182, rel=1e-9)
    assert float(e[5].mean()) == pytest.approx(30.23234578345288, rel=1e-9)


def test_scorer_protocol_constants_are_pinned():
    """A silent edit of these would make newly scored rows incomparable to the committed ones."""
    assert bj.DATASETS["kitti"] == dict(max_depth=80.0, min_depth=0.1, max_len=110, crop=(0, 374, 0, 1242), short_side=378,
                                        sky_mask="kitti_colour")
    assert bj.DATASETS["bonn"] == dict(max_depth=10.0, min_depth=0.1, max_len=110, crop=(0, 480, 0, 640), short_side=518)
    assert bj.DATASETS["vkitti"] == dict(max_depth=80.0, min_depth=0.1, max_len=64, crop=(0, 375, 0, 1242), short_side=378,
                                         sky_depth_m=655.0)
    assert (bj.LAG_KMAX, bj.LAG_T0) == (4, 4) and (bj.STATIC_LEN, bj.STATIC_T0) == (64, 8)
    assert (bj.GAIN_STAT, bj.GAIN_REF, bj.GAIN_MAX) == ("q99.5", 0.8796490759224356, 16.0)
    assert bj.OPTIN_DATASETS == ("vkitti",) and bj.DEFAULT_DATASETS == ["kitti", "bonn"]


def _vda_metric_py():
    import importlib.util
    p = Path(bj.PATHS["vda_repo"]) / "benchmark/eval/metric.py"
    if not p.is_file():
        pytest.skip("VDA benchmark/eval/metric.py not on this host")
    spec = importlib.util.spec_from_file_location("vda_metric_oracle", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_depth_errors_equal_vda_metric_py():
    """The oracle: score_clip's aligned and unaligned abs_rel / delta1 / rmse equal VDA's own metric.py on the same
    (clamped) depth, GT and valid mask, frames without a valid pixel dropped (as VDA's per-frame mean does)."""
    m = _vda_metric_py()
    depth, gt, r, d, c = _golden_inputs()
    out = bj.score_clip(depth, True, gt, r, d, c, 80.0, 0.1)
    g = torch.from_numpy(gt)
    valid = (g > 1e-3) & (g < 80.0)
    keep = valid.flatten(1).any(1)
    assert int(keep.sum()) == 7                                             # the fully invalid frame is dropped
    disp = (1.0 / torch.from_numpy(depth).clamp_min(1e-3)).clamp_min(1e-3)
    aligned = out["fit_scale"] * disp + out["fit_shift"]
    for tag, pred, v in (("aligned", (1.0 / aligned.clamp_min(1e-3)).clamp(1e-3, 80.0), valid),
                         ("unaligned", torch.from_numpy(depth).clamp(1e-3, 80.0), (g > 0.1) & (g < 80.0))):
        p, t, vm = pred[keep], g[keep], v[keep]
        assert out[f"abs_rel_{tag}"] == pytest.approx(float(m.abs_relative_difference(p, t, vm)), rel=1e-9)
        assert out[f"delta1_{tag}"] == pytest.approx(float(m.threshold_percentage(p, t, 1.25, vm)), abs=1e-9)
        assert out[f"rmse_{tag}"] == pytest.approx(float(m.rmse_linear(p, t, vm)), rel=1e-9)
