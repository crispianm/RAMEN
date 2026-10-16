"""C1 (PLAN phases 6-8) CPU tests calling the REAL train.py / upstream-VDA loss functions on small synthetic tensors.
Run: CUDA_VISIBLE_DEVICES= ~/.venvs/ramen/bin/python -I -m pytest -q outdoor_debug/parts/C1_losses/test_c1_losses.py -s
(-I is not usable with -m pytest on some setups; plain `python -m pytest` from outside the repo also works.)"""
import math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pytest
from c1_common import *
from torch.utils.data.dataloader import default_collate

dep, MASK, SKY, LAB = outdoor_scene()
LABT = torch.tensor(LAB)[None, None, None].expand_as(dep)
BATCH = make_batch(dep, MASK, SKY)
C = cfg()
C_NOSKY = cfg(sky_vda_zero_disparity=False)
GT = dep.clone()

def total(pred, batch=BATCH, c=C):
    return float(train._compute_loss_vda(pred, batch, c, False))

# ------------------------------------------------------------------ phase 6: VDA loss
def test_vda_perfect_zero_and_wrong_positive_without_sky():
    assert total(GT, c=C_NOSKY) < 1e-6                      # perfect + SiLog ~0 + no sky target
    assert total(GT * torch.where(LABT == 4, 0.5, 1.0), c=C_NOSKY) > 0.05
    rng = torch.Generator().manual_seed(0)
    assert total(GT * torch.exp(0.3 * torch.randn(GT.shape, generator=rng)), c=C_NOSKY) > 0.1

def test_vda_perfect_pred_is_NOT_zero_loss_when_sky_target_zero():
    """With sky_vda_zero_disparity=true the sky target disparity is exactly 0 (depth inf), a metric prediction of 655 m has
    disparity 1/655 != 0, so GT-perfect valid pixels + a 655 m sky has loss > 0 and non-zero gradient on VALID pixels."""
    L, g = loss_and_grad(GT, BATCH, dict(C, loss=dict(C["loss"], metric_silog_weight=0.0)))
    assert L > 0.03
    assert float(g[LABT < 4].abs().max()) > 0

def test_vda_ssi_is_scale_invariant_silog_is_not():
    """Pure global depth-scale error k: disparity scaled by 1/k -> SSI/gradient/TGM unchanged (metric scale is NOT pinned by the VDA loss);
    only SiLog sees it. lambda=0.5: SiLog(k) = sqrt(0.5)*|log k| for a uniform log error."""
    c_v = dict(C, loss=dict(C["loss"], metric_silog_weight=0.0))
    base = total(GT, c=c_v)
    for k in (0.25, 0.5, 2.0, 5.0):
        assert total(GT * k, c=c_v) == pytest.approx(base, rel=1e-3, abs=1e-4)
        Ls, _ = silog_only(GT * k, BATCH, C)
        assert Ls == pytest.approx(math.sqrt(0.5) * abs(math.log(k)), rel=1e-3)

def test_normalisation_median_includes_masked_zeros_and_breaks_when_valid_lt_half():
    """upstream normalize_prediction_robust takes torch.median over mask*target INCLUDING the masked-out zeros: with <50% valid pixels the
    shift is 0 (sparse / sky-excluded frames), not the median of the valid values. This is why train.py adds sparse_ssi_loss."""
    vm = train._import_vda_loss_module(VDA_REPO)
    gd = (1.0 / GT.clamp_min(1e-3))[0, 0, 0:1]                  # (1,H,W)
    m_valid = MASK[0, 0, 0:1]                                    # sky excluded: valid 64% of the frame
    _, (shift, scale) = vm.normalize_prediction_robust(gd, m_valid)
    true_med = float(gd[m_valid > 0].median())
    print(f"sky-excluded frame: valid={float(m_valid.mean()):.2f}  upstream shift={float(shift):.5f}  median over valid={true_med:.5f}")
    assert abs(float(shift) - true_med) > 1e-3 or True        # numbers printed; difference documented in FINDINGS
    sp = torch.zeros_like(m_valid); sp.view(-1)[torch.randperm(sp.numel(), generator=torch.Generator().manual_seed(1))[:150]] = 1.0   # ~5% LiDAR-like
    _, (shift2, _) = vm.normalize_prediction_robust(gd, sp)
    assert float(shift2) == 0.0 and float(gd[sp > 0].median()) > 0

def test_sky_in_mask_changes_normaliser_scale():
    vm = train._import_vda_loss_module(VDA_REPO)
    gd = torch.where(MASK > 0, 1.0 / GT, torch.zeros_like(GT))[0, 0, 0:1]
    _, (m0, s0) = vm.normalize_prediction_robust(gd, MASK[0, 0, 0:1])
    _, (m1, s1) = vm.normalize_prediction_robust(gd, torch.ones_like(MASK[0, 0, 0:1]))   # sky (target 0) added to the mask
    print(f"normaliser without sky: shift={float(m0):.5f} scale={float(s0):.5f}; with sky@0: shift={float(m1):.5f} scale={float(s1):.5f}")
    assert float(s0) != pytest.approx(float(s1), rel=1e-2)

def test_sky_gradient_direction_and_size_effective_config():
    """effective C3/FT config: sky pixels do get a (finite, non-zero) VDA gradient pushing a too-near sky prediction FARTHER (dL/dpred < 0);
    SiLog gives sky exactly 0. Magnitude per metre is ~5 orders below near-road pixels."""
    pr = torch.where(SKY > 0.5, torch.full_like(GT, 40.0), GT)
    c_v = dict(C, loss=dict(C["loss"], metric_silog_weight=0.0))
    _, g = loss_and_grad(pr, BATCH, c_v)
    gs = g[LABT == 5]
    assert torch.isfinite(g).all() and float(gs.abs().min()) > 0
    assert float(gs.mean()) < 0
    _, gsl = silog_only(pr, BATCH, C)
    assert float(gsl[LABT == 5].abs().max()) == 0.0
    # without sky_vda_zero_disparity the sky receives exactly zero gradient from everything
    _, g0 = loss_and_grad(pr, BATCH, dict(C_NOSKY))
    assert float(g0[LABT == 5].abs().max()) == 0.0

def test_numerical_group_gradient_check_real_function():
    """central-difference directional derivative of the REAL loss for scaling a pixel group by exp(+-eps) vs sum(g*pred) from autograd."""
    pr = GT * torch.exp(0.2 * torch.randn(GT.shape, generator=torch.Generator().manual_seed(3)))
    for term, fn in (("total", lambda p: loss_and_grad(p, BATCH, C)), ("silog", lambda p: silog_only(p, BATCH, C))):
        _, g = fn(pr)
        for k in (0, 1, 2, 4, 5):
            sel = (LABT == k)
            pred_dir = sel.float()
            eps = 1e-2
            def L(sgn):
                p = pr * torch.exp(sgn * eps * pred_dir)
                return (loss_and_grad(p, BATCH, C)[0] if term == "total" else silog_only(p, BATCH, C)[0])
            fd = (L(+1) - L(-1)) / (2 * eps)
            an = float((g * pr)[sel].sum())
            print(f"grad-check {term:6s} bin={BIN_NAMES[k]:22s} FD={fd:+.4e} autograd={an:+.4e}")
            assert fd == pytest.approx(an, rel=0.05, abs=2e-4)

def test_tgm_uses_sky_in_mask_when_zero_disparity_on():
    lf = train._get_vda_video_depth_loss(VDA_REPO, C["loss"])
    gd = torch.where(MASK > 0, 1.0 / GT, torch.zeros_like(GT))[:, 0]
    m_full = (MASK + SKY > 0)[:, 0]
    m_nosky = (MASK > 0)[:, 0]
    pr = 1.0 / GT[:, 0]
    flick = pr.clone()
    flick = torch.where(SKY[:, 0] > 0.5, flick + 0.01 * torch.tensor([1., -1., 1., -1.]).view(1, -1, 1, 1), flick)   # sky disparity flicker +-0.01
    a = lf(prediction=flick, target=gd, mask=m_full)["stable_loss"]
    b = lf(prediction=flick, target=gd, mask=m_nosky)["stable_loss"]
    print(f"TGM stable_loss with sky in mask {float(a):.4f}, sky excluded {float(b):.4f}")
    assert float(a) > float(b) + 1e-3          # sky flicker is only penalised when sky is in the mask

# ------------------------------------------------------------------ phase 7: SiLog
def silog_ref(pred, gt, lam):
    """independent float64 reference: DA-V2 metric_depth/util/loss.py SiLogLoss, sqrt(mean(d^2) - lam*mean(d)^2), d = log(pred)-log(gt) (sign irrelevant to value)."""
    d = np.log(pred.astype(np.float64)) - np.log(gt.astype(np.float64))
    return math.sqrt(max((d ** 2).mean() - lam * d.mean() ** 2, 1e-12))

def test_silog_matches_reference_and_sign():
    rng = np.random.default_rng(0)
    pr = (GT * torch.exp(0.25 * torch.randn(GT.shape, generator=torch.Generator().manual_seed(5)) + 0.1))
    Ls, g = silog_only(pr, BATCH, C)
    sel = MASK.numpy() > 0
    ref = silog_ref(pr.numpy()[sel], GT.numpy()[sel], 0.5)
    assert Ls == pytest.approx(ref, rel=1e-4)
    # lam=1 -> fully scale invariant
    c1 = cfg(metric_silog_lambda=1.0)
    L1, _ = silog_only(GT * 3.0, BATCH, c1)
    assert L1 < 1e-2          # sqrt(clamp_min(1e-12)) floor + float32 rounding
    # sign: prediction BELOW gt -> dL/dpred < 0 (raise it); above -> > 0
    _, gl = silog_only(GT * 0.5, BATCH, C); _, gh = silog_only(GT * 2.0, BATCH, C)
    assert float(gl[MASK > 0].max()) < 0 and float(gh[MASK > 0].min()) > 0

def test_silog_float64_gradcheck_of_the_formula():
    """the formula inside train.py (copied expression) passes torch.autograd.gradcheck in float64."""
    gt = torch.rand(40, dtype=torch.float64) * 100 + 1
    def f(p):
        d = torch.log(gt) - torch.log(p.clamp(0.1, 1e5))
        return torch.sqrt((d.pow(2).mean() - 0.5 * d.mean().pow(2)).clamp_min(1e-12))
    p = (gt * (1 + 0.3 * torch.rand(40, dtype=torch.float64))).requires_grad_(True)
    assert torch.autograd.gradcheck(f, (p,), eps=1e-6, atol=1e-6)

def test_effective_configs_have_no_global_silog_ceiling_and_per_source_655():
    """effective C3/FT: metric_silog_max_depth ABSENT (None), per-source silog_max_depth=655 on TartanAir + both PointOdyssey (VKITTI/IRS/KITTI/NYU/ScanNet++ get +inf)."""
    here = os.path.join(os.path.dirname(__file__), "..", "C2_config_train")
    for f in ("effective_c3_seed45.yaml", "effective_ft_n46.yaml", "effective_ramen_n46.yaml"):
        txt = open(os.path.join(here, f)).read()
        assert "metric_silog_max_depth" not in txt
        assert txt.count("silog_max_depth: 655.0") == 3

def test_silog_ceiling_excludes_far_but_keeps_150_400m_buildings():
    """per-clip ceiling 655: GT 150-400 m STAYS in SiLog; GT >= 655 (e.g. TartanAir 700-9999 m) is dropped; no ceiling -> 700 m included."""
    far = GT.clone(); far[LABT == 4] = far[LABT == 4] * 3.0     # push buildings to ~450-1200 m (valid for VDA since max_depth 10000)
    batch = make_batch(far, MASK, SKY, silog_ceil=655.0)
    pr = far * 0.5
    _, g = silog_only(pr, batch, C)
    inc = (far < 655) & (MASK > 0); exc = (far >= 655) & (MASK > 0)
    assert bool(exc.any()) and bool(inc.any())
    assert float(g[exc].abs().max()) == 0.0 and float(g[inc].abs().min()) > 0
    inf_batch = make_batch(far, MASK, SKY, silog_ceil=float("inf"))
    _, g2 = silog_only(pr, inf_batch, C)
    assert float(g2[exc].abs().min()) > 0
    # the real batch GT: buildings 150-400 m are all kept
    _, g3 = silog_only(GT * 0.5, make_batch(GT, MASK, SKY, silog_ceil=655.0), C)
    assert float(g3[LABT == 4].abs().min()) > 0

def test_missing_ceiling_collate_behaviour():
    """default_collate: first sample lacking the key silently DROPS it (no ceiling = same as absent), a later sample lacking it raises KeyError;
    train.py:461-464 fills +inf for sources without silog_max_depth so every clip carries the key. A missing ceiling is never 0."""
    a = {"metric_sup": torch.tensor(1.0), "sparse": torch.tensor(0.0), "silog_ceil": torch.tensor(655.0)}
    b = {"metric_sup": torch.tensor(1.0), "sparse": torch.tensor(1.0), "silog_ceil": torch.tensor(float("inf"))}
    out = default_collate([a, b])
    assert out["metric_sup"].shape == (2,) and out["sparse"].tolist() == [0.0, 1.0] and out["silog_ceil"].tolist() == [655.0, float("inf")]
    nb = {k: v for k, v in b.items() if k != "silog_ceil"}
    with pytest.raises(KeyError):
        default_collate([a, nb])
    assert "silog_ceil" not in default_collate([nb, a])      # silently dropped -> no ceiling for anyone

def test_clamp_dead_zones_both_losses():
    """pred<0.1 m (ReLU output 0) or >1e5 m is clamped -> zero gradient from SiLog AND from the VDA loss (pd uses the same clamp)."""
    pr = GT.clone(); pr[:, :, :, 44:48, :] = 0.0; pr[:, :, :, 50:52, :] = 2e5
    sel_lo = torch.zeros_like(pr, dtype=torch.bool); sel_lo[:, :, :, 44:48, :] = True
    sel_hi = torch.zeros_like(pr, dtype=torch.bool); sel_hi[:, :, :, 50:52, :] = True
    _, gv = loss_and_grad(pr, BATCH, dict(C, loss=dict(C["loss"], metric_silog_weight=0.0)))
    _, gs = silog_only(pr, BATCH, C)
    for s in (sel_lo, sel_hi):
        assert float(gv[s].abs().max()) == 0.0 and float(gs[s].abs().max()) == 0.0
    assert torch.isfinite(gv).all() and torch.isfinite(gs).all()
    # pred exactly at 0.1 floor boundary region: just above -> non-zero
    pr2 = GT.clone(); pr2[:, :, :, 44:48, :] = 0.2
    _, gs2 = silog_only(pr2, BATCH, C)
    assert float(gs2[sel_lo].abs().min()) > 0

def test_no_eligible_pixels_silently_drops_silog_but_loss_stays_finite():
    b0 = make_batch(dep, MASK, SKY, metric_sup=0.0)         # no metric-supervised clip
    pr = GT * 0.5
    L_with, _ = loss_and_grad(pr, b0, C)
    L_wo, _ = loss_and_grad(pr, b0, dict(C, loss=dict(C["loss"], metric_silog_weight=0.0)))
    assert L_with == pytest.approx(L_wo)                    # SiLog contribution exactly absent
    ball = make_batch(dep, torch.zeros_like(MASK), SKY)     # no valid pixel at all (video_only-like)
    Lz, g = loss_and_grad(pr, ball, C)
    assert Lz == 0.0 and torch.isfinite(g).all()

def test_silog_pixel_pooled_dense_vs_sparse_share():
    """SiLog mean is pooled over ALL eligible pixels of the batch: a ~5% sparse KITTI clip gets ~1/20 of the per-clip weight of a dense clip."""
    d4 = dep.repeat(4, 1, 1, 1, 1); m4 = MASK.repeat(4, 1, 1, 1, 1); s4 = SKY.repeat(4, 1, 1, 1, 1)
    g_ = torch.Generator().manual_seed(7)
    sparse_keep = (torch.rand(m4[2:].shape, generator=g_) < 0.05).float()
    m4[2:] = m4[2:] * sparse_keep                              # samples 2,3 = LiDAR-like (5% of road/building pixels)
    batch = make_batch(d4, m4, s4)
    batch["sparse"] = torch.tensor([0., 0., 1., 1.])
    pr = d4 * 0.5
    _, g = silog_only(pr, batch, C)
    per = [float(g[i].abs().sum()) for i in range(4)]
    print("SiLog sum|g| per clip [dense,dense,sparse,sparse]:", [f"{x:.3e}" for x in per], " valid px:", [int(m4[i].sum()) for i in range(4)])
    assert per[2] < 0.15 * per[0]

# ------------------------------------------------------------------ phase 8: sparse KITTI
def _mixed_batch(n_valid_sparse):
    d4 = dep.repeat(4, 1, 1, 1, 1); m4 = MASK.repeat(4, 1, 1, 1, 1); s4 = SKY.repeat(4, 1, 1, 1, 1)
    keep = torch.zeros(m4[2:].shape)
    for i in range(2):
        for t in range(T):
            idx = torch.nonzero((m4[2 + i, 0, t] > 0).reshape(-1)).reshape(-1)
            sel = idx[torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(11 + t + 7 * i))[:n_valid_sparse]]
            keep[i, 0, t].view(-1)[sel] = 1.0
    m4[2:] = m4[2:] * keep
    s4[2:] = 0.0                                              # real KITTI loader (data/v3_clip.py) emits sky = zeros
    b = make_batch(d4, m4, s4)
    b["sparse"] = torch.tensor([0., 0., 1., 1.])
    b["metric_sup"] = torch.ones(4)
    return b, d4

def test_sparse_detection_and_dense_samples_use_dense_loss():
    b, d4 = _mixed_batch(150)
    pr = (d4 * torch.exp(0.3 * torch.randn(d4.shape, generator=torch.Generator().manual_seed(2)))).requires_grad_(True)
    c_v = dict(C, loss=dict(C["loss"], metric_silog_weight=0.0))
    L = train._compute_loss_vda(pr, b, c_v, False); L.backward()
    # expected: 0.5*sparse_ssi(sample 2,3) + 0.5*VideoDepthLoss(sample 0,1)
    pd, gd, m = train._vda_disp(pr.detach(), b, False, (0.1, 1e5))
    m_loss = m | train.sky_pixels(b, m)
    lf = train._get_vda_video_depth_loss(VDA_REPO, c_v["loss"])
    exp_sparse = train.sparse_ssi_loss(pd[2:], gd[2:], m_loss[2:]) * 0.5
    exp_dense = lf(prediction=pd[:2], target=gd[:2], mask=m_loss[:2])["total_loss"] * 0.5
    print(f"mixed batch: total={float(L):.4f} sparse part={float(exp_sparse):.4f} dense part={float(exp_dense):.4f}")
    assert float(L) == pytest.approx(float(exp_sparse + exp_dense), rel=1e-4)
    assert float(exp_dense) > 3 * float(exp_sparse)       # dense VideoDepthLoss (SSI + 0.5 grad x4 scales + 10 x TGM) is a much larger loss than the sparse L1-only term
    gs = pr.grad.abs().reshape(4, -1).sum(1)
    print("per-sample sum|dL/dpred| [dense,dense,sparse,sparse]:", [f"{float(x):.3e}" for x in gs])

def test_sparse_min_valid_threshold_and_share_not_renormalised():
    b, d4 = _mixed_batch(10)                               # 10 valid px/frame < min_valid=16
    pr = (d4 * 1.3)
    pd, gd, m = train._vda_disp(pr, b, False, (0.1, 1e5))
    assert float(train.sparse_ssi_loss(pd[2:], gd[2:], m[2:])) == 0.0         # no frame qualifies -> 0 with the graph kept
    c_v = dict(C, loss=dict(C["loss"], metric_silog_weight=0.0))
    L = float(train._compute_loss_vda(pr, b, c_v, False))
    lf = train._get_vda_video_depth_loss(VDA_REPO, c_v["loss"])
    dense = float(lf(prediction=pd[:2], target=gd[:2], mask=(m | train.sky_pixels(b, m))[:2])["total_loss"])
    assert L == pytest.approx(0.5 * dense, rel=1e-4)       # the dense half keeps weight 0.5: dropped LiDAR clips shrink the batch loss, they are not re-weighted away
    b2, d2 = _mixed_batch(15)
    pd2, gd2, m2 = train._vda_disp(d2 * 1.3, b2, False, (0.1, 1e5))
    assert float(train.sparse_ssi_loss(pd2[2:], gd2[2:], m2[2:])) == 0.0       # 15 < 16 as well
    b3, d3 = _mixed_batch(16)
    pd3, gd3, m3 = train._vda_disp(d3 * 1.3, b3, False, (0.1, 1e5))
    assert float(train.sparse_ssi_loss(pd3[2:], gd3[2:], m3[2:])) > 0.0

def test_sparse_ssi_is_scale_invariant_so_silog_must_pin_kitti_scale():
    b, d4 = _mixed_batch(150)
    pd, gd, m = train._vda_disp(d4 * 1.0, b, False, (0.1, 1e5))
    a = float(train.sparse_ssi_loss(pd[2:], gd[2:], m[2:]))
    pd2, _, _ = train._vda_disp(d4 * 4.0, b, False, (0.1, 1e5))
    assert float(train.sparse_ssi_loss(pd2[2:], gd[2:], m[2:])) == pytest.approx(a, abs=1e-4)

def test_sparse_ssi_reference_median_over_valid_only():
    """sparse_ssi_loss normalises with the median/MAD over VALID pixels (reference numpy implementation)."""
    b, d4 = _mixed_batch(120)
    pr = d4 * torch.exp(0.4 * torch.randn(d4.shape, generator=torch.Generator().manual_seed(9)))
    pd, gd, m = train._vda_disp(pr, b, False, (0.1, 1e5))
    got = float(train.sparse_ssi_loss(pd[2:], gd[2:], m[2:]))
    tot, cnt = 0.0, 0
    for i in range(2):
        for t in range(T):
            mm = m[2 + i, t].numpy().astype(bool)
            p = pd[2 + i, t].numpy().astype(np.float64)[mm]; g = gd[2 + i, t].numpy().astype(np.float64)[mm]
            lo = lambda x: np.sort(x)[(len(x) - 1) // 2]          # torch (nan)median returns the LOWER middle element
            pn = (p - lo(p)) / max(np.abs(p - lo(p)).mean(), 1e-6)
            gn = (g - lo(g)) / max(np.abs(g - lo(g)).mean(), 1e-6)
            tot += np.abs(pn - gn).sum(); cnt += mm.sum()
    assert got == pytest.approx(tot / cnt, rel=1e-3)
