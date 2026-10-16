"""Decode real train frames per source with the REAL dataset classes (degrade off); raw-decoded stats + processed stats + panels."""
import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
import torch, json, time
from PIL import Image
from data.vda_semantics_dataset import (VDASemanticsTartanAirDataset, VDASemanticsVKITTIDataset, VDASemanticsPointOdysseyDataset, VDASemanticsIRSDataset)
from data.video_clip_dataset import _decode_depth
from data.vkitti_dataset import _decode_vkitti_depth
from data.pointodyssey_dataset import _PO_DEPTH_SCALE
from data.kitti_raw_train import KITTIRawTrainDataset, read_kitti_gt
from data.spring_dataset import SpringDataset
import yaml
random.seed(0); np.random.seed(0)
which = sys.argv[1:]
common = dict(num_frames=32, image_size=(518, 518), split="train", clip_stride=32, frame_stride=1, seed=45, degrade=False)

def make(name):
    if name == "tartanair": return VDASemanticsTartanAirDataset(root=f"{T}/tartanair", val_fraction=0.1, max_depth=10000.0, **common)
    if name == "vkitti": 
        ds = VDASemanticsVKITTIDataset(root=f"{T}/vkitti", variants=["clone","fog","morning","overcast","rain","sunset","15-deg-left","15-deg-right","30-deg-left","30-deg-right"], cameras=["Camera_0","Camera_1"], no_holdout=True, max_depth=80.0, **common)
        ds.vkitti_max_depth = 655.0; return ds
    if name == "pointodyssey": return VDASemanticsPointOdysseyDataset(root=f"{T}/point_odyssey/train", keep_list=REPO+"/provenance/dn01_vdamix/po_filter_v1.json", keep_sha256="ff4851c6ad1d8bb0754ef9ef30aeb1c83562624041074acd41e41b5e9eedb34c", max_depth=999.0, **common)
    if name == "kitti_raw": return KITTIRawTrainDataset(root=f"{T}/kitti_raw", gt_root=f"{T}/kitti_raw_gt", allow_missing=False, max_depth=80.0, **common)
    if name == "irs": return VDASemanticsIRSDataset(root=f"{T}/irs", val_fraction=0.0, max_depth=80.0, **common)
    if name == "spring": return SpringDataset(root=f"{T}/spring", val_fraction=0.03, max_depth=80.0, **common)

def raw_depth(name, ds, clip, i):
    files = ds.clips[clip]
    dep = files[1][i]
    if name == "tartanair": return _decode_depth(str(dep))
    if name == "vkitti": return _decode_vkitti_depth(str(dep))
    if name == "pointodyssey": return cv2.imread(str(dep), cv2.IMREAD_UNCHANGED).astype(np.float32) * _PO_DEPTH_SCALE
    if name == "kitti_raw": return read_kitti_gt(dep)
    if name == "irs": 
        from data.irs_dataset import _read_exr_depth; return _read_exr_depth(dep)
    if name == "spring": return ds.read_depth(dep)

rows = []
for name in which:
    t0 = time.time(); ds = make(name); print(name, "clips", len(ds), "build s", round(time.time()-t0,1), flush=True)
    rng = random.Random(1)
    nclip = 16
    cidx = rng.sample(range(len(ds)), nclip)
    panels = []
    for k, c in enumerate(cidx):
        s = ds[c]
        fr, de, ma, sk = s["frames"], s["depth"], s["mask"], s["sky"]
        # raw stats for frame 0 and middle frame of clip (ORIGINAL decoded depth)
        start = ds.clips[c][3]
        for t in (0, 16):
            d = raw_depth(name, ds, c, start + t)
            st = raw_stats(d); st.update(source=name, clip=int(c), video=str(s["video_id"]), frame=start + t, kind="raw", raw_hw=f"{d.shape[0]}x{d.shape[1]}")
            rows.append(st)
            m = ma[0, t].numpy() > 0.5
            dd = de[0, t].numpy()
            p = dict(source=name, clip=int(c), video=str(s["video_id"]), frame=start + t, kind="processed", n=int(m.size), f_valid=float(m.mean()), f_sky=float((sk[0, t] > .5).float().mean()),
                     f_invalid_nonsky=float(((~m) & (sk[0, t].numpy() < .5)).mean()), f_valid_gt80=float((m & (dd > 80)).mean()), f_valid_gt200=float((m & (dd > 200)).mean()),
                     f_valid_gt655=float((m & (dd > 655)).mean()), f_depth_ge_cap=float((dd >= ds.max_depth - 1e-3).mean()))
            p.update(pct(dd[m])); rows.append(p)
        if k < 8:
            t = 0
            rgb = (fr[:, t].permute(1, 2, 0).numpy() * (255 if fr.dtype != torch.uint8 else 1)).clip(0, 255).astype(np.uint8)
            m = ma[0, t].numpy() > 0.5; sky = sk[0, t].numpy() > 0.5; dd = de[0, t].numpy()
            col = color_depth(dd, m | sky)
            col[sky & ~m] = (255, 255, 255)
            mv = np.repeat((m * 255).astype(np.uint8)[..., None], 3, 2)
            sv = np.repeat((sky * 255).astype(np.uint8)[..., None], 3, 2)
            ov = rgb.copy().astype(np.float32)
            ov[sky] = 0.5 * ov[sky] + 0.5 * np.array([255, 0, 0]); ov[~m & ~sky] = 0.5 * ov[~m & ~sky] + 0.5 * np.array([0, 0, 255]); ov[m] = 0.7 * ov[m] + 0.3 * col[m]
            panels.append(np.concatenate([rgb, col, mv, sv, ov.astype(np.uint8)], 1))
            if k < 2:   # temporal strip: frames 0,10,20,31 RGB / depth, plus consecutive vs random-pair RGB difference
                strip = []
                for t in (0, 10, 20, 31):
                    r = (fr[:, t].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    mm = ma[0, t].numpy() > .5; ss = sk[0, t].numpy() > .5
                    cc = color_depth(de[0, t].numpy(), mm | ss); cc[ss & ~mm] = 255
                    strip.append(np.concatenate([r, cc], 0))
                Image.fromarray(np.concatenate(strip, 1)).resize((4 * 259, 2 * 259)).save(f"{OUT}/{name}_temporal_strip_{k}.png")
            cons = float(np.mean([np.abs(fr[:, t+1] - fr[:, t]).mean().item() for t in range(31)]))
            other = ds[(c + 7) % len(ds)]["frames"]
            rnd = float(np.mean([np.abs(other[:, t] - fr[:, t]).mean().item() for t in range(32)]))
            rows.append(dict(source=name, clip=int(c), video=str(s["video_id"]), kind="temporal", rgb_consec_mad=cons, rgb_otherclip_mad=rnd,
                             depth_consec_mad=float(np.mean([np.abs(de[:, t+1] - de[:, t]).mean().item() for t in range(31)]))))
    img = np.concatenate(panels, 0)
    Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2)).save(f"{OUT}/{name}_panels.png")
    print(name, "done", flush=True)
import pandas as pd
pd.DataFrame(rows).to_csv(f"{OUT}/stats_{'_'.join(which)}.csv", index=False)
