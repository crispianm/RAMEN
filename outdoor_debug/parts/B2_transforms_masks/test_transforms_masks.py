"""CPU regression/evidence tests for PLAN phase 4/5 on the REAL dataset classes (data/vda_semantics_dataset.py, data/vda_resize.py).
Synthetic files are written in the real on-disk encodings (VKITTI uint16 cm PNG, TartanAir float32-in-BGRA PNG) and read back through the real __getitem__.
Run: python -I outdoor_debug/parts/B2_transforms_masks/test_transforms_masks.py   (from the repo root; also collected by pytest)"""
import sys, random, tempfile, pathlib
sys.path.insert(0, '.')
import numpy as np, cv2, torch
from PIL import Image
from data.vda_resize import vda_resize_depth_like, vda_resize_size, vda_crop_box
from data import vda_semantics_dataset as V

H0, W0 = 375, 1242            # KITTI / VKITTI native size
S = 518

def _scene(tmp):
    """RGB encodes the column (R = x*255/(W-1)) and row (G); depth encodes sky (65535), holes (0), a near wedge, a far (600 m) band and a 'cliff' column pair."""
    tmp = pathlib.Path(tmp)
    rgb = np.zeros((H0, W0, 3), np.uint8)
    xs = np.arange(W0)
    rgb[..., 0] = (xs[None, :] * 255 // (W0 - 1)).astype(np.uint8)
    rgb[..., 1] = (np.arange(H0)[:, None] * 255 // (H0 - 1)).astype(np.uint8)
    cm = np.full((H0, W0), 2000, np.uint16)            # 20 m
    cm[:120, :] = 65535                                  # sky sentinel (655.35 m) on the top 120 rows
    cm[120:140, :] = 60000                               # a 600 m far band directly under the sky (distant valid geometry next to the sentinel)
    cm[200:210, 300:600] = 0                             # zero holes (invalid)
    cm[300:, :] = 500                                    # 5 m ground
    Image.fromarray(rgb).save(tmp / 'rgb_0.png'); cv2.imwrite(str(tmp / 'd_0.png'), cm)
    return str(tmp / 'rgb_0.png'), str(tmp / 'd_0.png'), cm

def _vk(cls, tmp, split='train', **kw):
    rgb, dep, cm = _scene(tmp)
    ds = object.__new__(cls)
    ds.num_frames, ds.image_size, ds.split, ds.frame_stride = 2, (S, S), split, 1
    ds.clips = [([rgb, rgb], [dep, dep], 'Scene99', 0)]
    ds.vkitti_max_depth = kw.get('vkitti_max_depth')
    ds.degrade, ds.degrade_mode, ds.degrade_severity, ds.seed = kw.get('degrade', False), ['awgn_blur'], 1.0, 0
    ds.severity_max = torch.ones(1)
    return ds, cm

def test_nearest_never_mixes_values():
    cm = np.full((H0, W0), 20.0, np.float32); cm[:120] = 655.35; cm[120:140] = 600.0; cm[200:210, 300:600] = 0.0
    nw, nh = vda_resize_size(W0, H0, S)
    out = vda_resize_depth_like(cm, nw, nh)
    assert set(np.unique(out).tolist()) <= set(np.unique(cm).tolist()), "nearest introduced a new depth value"
    bil = cv2.resize(cm, (nw, nh), interpolation=cv2.INTER_LINEAR)       # COUNTERFACTUAL (not used by the pipeline)
    mixed = int((~np.isin(bil, np.unique(cm))).sum())
    inv_mix = int(((bil > 0) & (bil < 20.0 - 1e-3)).sum())
    print(f"nearest: new values=0 ; counterfactual bilinear: {mixed} px with values not in the source ({100*mixed/bil.size:.3f}%), {inv_mix} px strictly between a hole (0) and 20 m")
    assert mixed > 0   # shows the test can detect mixing

def test_vkitti_real_getitem_masks_and_alignment():
    with tempfile.TemporaryDirectory() as t:
        ds, cm = _vk(V.VDASemanticsVKITTIDataset, t)
        random.seed(3); it = ds[0]
        f, d, m, s = it['frames'], it['depth'], it['mask'], it['sky']
        assert f.shape == (3, 2, S, S) and d.shape == m.shape == s.shape == (1, 2, S, S)
        assert set(torch.unique(m).tolist()) <= {0.0, 1.0} and set(torch.unique(s).tolist()) <= {0.0, 1.0}
        valid, sky = m > 0.5, s > 0.5
        assert not bool((valid & sky).any())                      # disjoint
        assert float(d[valid].max()) < 655.0 and float(d[valid].min()) > 0
        assert torch.isfinite(d).all()
        assert float(d[sky].min()) == 655.0 and float(d[sky].max()) == 655.0   # sentinel clipped to 655.0, labelled sky, NOT valid
        assert float(d[~valid & ~sky].abs().max()) == 0.0         # holes stay exactly 0
        # far band (600 m) is VALID geometry and not sky; holes are invalid but not sky
        assert bool(((d > 599) & (d < 601))[valid].any())
        # RGB<->depth alignment: R channel encodes the original column, depth rows encode the structure; compare edges
        # sky/valid boundary row in RGB-G coordinates: G encodes original row; sky rows are orig rows <120 -> G < 120*255/374
        g = f[1][0]                                                # (S,S) green = original row/374 (cubic-resized)
        row_orig = g * 374.0
        sky_rows = row_orig[sky[0, 0]]
        assert float(sky_rows.max()) < 120 + 4, float(sky_rows.max())  # every sky pixel's source row < ~120 (+interp tolerance)
        nonsky_rows = row_orig[valid[0, 0]]
        assert float(nonsky_rows.min()) > 120 - 4, float(nonsky_rows.min())
        # same crop for both frames (identical input frames -> identical output)
        assert torch.equal(d[:, 0], d[:, 1]) and torch.equal(f[:, 0], f[:, 1])
        print("vkitti: pct valid %.1f sky %.1f ; valid&sky 0 ; sentinel->655.0 under sky ; holes 0" % (100*valid.float().mean(), 100*sky.float().mean()))

def test_vkitti_max_depth_cap_does_not_make_far_pixels_sky():
    with tempfile.TemporaryDirectory() as t:
        ds, _ = _vk(V.VDASemanticsVKITTIDataset, t, vkitti_max_depth=80)
        random.seed(3); it = ds['0' and 0]
        d, m, s = it['depth'], it['mask'] > .5, it['sky'] > .5
        far = (d > 80) & (d < 655)
        assert bool(far.any()) and not bool((far & m).any()) and not bool((far & s).any())   # capped-out far geometry is neither valid nor sky (silently dropped)

def test_photometric_and_static_touch_only_rgb():
    with tempfile.TemporaryDirectory() as t:
        for split in ('train',):
            a, _ = _vk(V.VDASemanticsDegradedVKITTIDataset, t, degrade=False)
            b, _ = _vk(V.VDASemanticsDegradedVKITTIDataset, t, degrade=True)
            random.seed(11); ia = a[0]; random.seed(11); ib = b[0]
            for k in ('depth', 'mask', 'sky'):
                assert torch.equal(ia[k], ib[k]), k
            assert 'degraded_frames' in ib and float((ib['degraded_frames'] - ib['frames']).abs().mean()) > 1e-3

def _ta_write(tmp):
    tmp = pathlib.Path(tmp)
    rgb = np.full((480, 640, 3), 120, np.uint8); rgb[:, :, 0] = (np.arange(640) * 255 // 639).astype(np.uint8)
    d = np.full((480, 640), 15.0, np.float32)
    d[:100] = np.inf; d[100:110] = 2e4; d[110:120] = 10000.0; d[120:130] = 9999.0; d[130:140] = 700.0; d[140:150] = np.nan; d[150:160] = 0.0; d[160:170] = -1.0
    Image.fromarray(rgb).save(tmp / 'r.png')
    cv2.imwrite(str(tmp / 'd.png'), np.ascontiguousarray(d).view(np.uint8).reshape(480, 640, 4))
    return str(tmp / 'r.png'), str(tmp / 'd.png'), d

def test_tartanair_sentinels_through_real_getitem():
    with tempfile.TemporaryDirectory() as t:
        rgb, dep, d = _ta_write(t)
        for md in (80.0, 10000.0):                                    # 80 = data.max_depth default, 10000 = the C3 TartanAir spec
            ds = object.__new__(V.VDASemanticsTartanAirDataset)
            ds.num_frames, ds.image_size, ds.split, ds.frame_stride, ds.max_depth = 1, (S, S), 'val', 1, md   # val = centre crop (deterministic)
            ds.clips = [([rgb], [dep], 'E/P', 0)]
            it = ds[0]
            D, M, K = it['depth'][0, 0], it['mask'][0, 0] > .5, it['sky'][0, 0] > .5
            assert torch.isfinite(D).all() and not bool((M & K).any())
            assert set(torch.unique(it['mask']).tolist()) <= {0., 1.} and set(torch.unique(it['sky']).tolist()) <= {0., 1.}
            assert float(D[M].max()) < md and float(D[M].min()) > 0
            assert float(D.max()) <= md                               # everything clipped to max_depth: inf sky is stored AS max_depth
            # rows (in the 480->518x? resized then cropped frame): classify by source row via a nearest probe
            nw, nh = vda_resize_size(640, 480, S)
            x0, y0 = vda_crop_box(nw, nh, S, 'val')
            src_row = lambda r: int(np.clip((r + y0 + .5) * 480 / nh, 0, 479))
            classes = {}
            for r in range(S):
                sr = src_row(r)
                if not 3 <= sr % 10 <= 6: continue   # probe only the middle of each 10-row band (nearest rounding at band edges)
                key = ('sky' if bool(K[r, 5]) else 'valid' if bool(M[r, 5]) else 'neither')
                classes.setdefault((sr // 10 * 10), set()).add(key)
            print(f"TA max_depth={md}:", {k: sorted(v) for k, v in sorted(classes.items()) if k in (0, 100, 110, 120, 130, 140, 150, 160)})
            if md == 10000.0:
                assert classes[110] == {'neither'}, classes[110]      # EXACTLY 10000 m: not valid (strict <), not sky (strict >): a hole in the sky label
                assert classes[120] == {'valid'}
            assert classes[100] == {'sky'} and classes[0] == {'sky'}
            assert classes[150] == {'neither'} and classes[160] == {'neither'}   # 0 / negative: invalid, never sky
            # FINDING B2-03: NaN depth is labelled SKY (data/vda_semantics_dataset.py:101 `sky = ~np.isfinite(d_raw) | ...` does not separate NaN from +inf)
            assert 'sky' in classes[140], classes[140]
            if md == 80.0:
                assert classes[130] == {'neither'}                    # 700 m with the 80 m cap: neither valid nor sky (silently dropped)
            else:
                assert classes[130] == {'valid'}

if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('PASS', n)
