import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
from data.kitti_raw_train import KITTIRawTrainDataset, read_kitti_gt
from data.vda_resize import vda_resize_size, vda_resize_depth_like
import json
ds = KITTIRawTrainDataset(root=f"{T}/kitti_raw", gt_root=f"{T}/kitti_raw_gt", num_frames=32, image_size=(518,518), split="train", clip_stride=32, max_depth=80.0, seed=45)
rng = random.Random(2); mx = []; zero = []; nv = []; rowcov = np.zeros(10); allv = []; gt80 = []; nvp = 0
raw_png_dtypes = set(); n_after = []; n_raw = []
for i in rng.sample(range(len(ds)), 300):
    c = ds.clips[i]; f = c[1][c[3] + rng.randrange(32)]
    a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED); raw_png_dtypes.add(str(a.dtype))
    d = a.astype(np.float32) / 256.0
    v = d > 0; mx.append(d.max()); zero.append(1 - v.mean()); nv.append(v.sum()); allv.append(d[v]); gt80.append((d > 80).mean())
    H = d.shape[0]
    for r in range(10): rowcov[r] += v[r * H // 10:(r + 1) * H // 10].mean()
    # resize: valid points before vs after (nearest, full resized frame, before the crop)
    rw, rh = vda_resize_size(d.shape[1], d.shape[0], 518)
    vr = vda_resize_depth_like(((d > 0) & (d < 80)).astype(np.float32), rw, rh)
    n_raw.append(int(((d > 0) & (d < 80)).sum())); n_after.append(float(vr.sum()))
allv = np.concatenate(allv)
res = dict(frames=300, png_dtypes=sorted(raw_png_dtypes), max_depth_m=float(np.max(mx)), frames_with_max_gt80=float(np.mean(np.array(mx) > 80)), p999_valid=float(np.percentile(allv, 99.9)),
           mean_zero_frac=float(np.mean(zero)), mean_valid_frac=float(1 - np.mean(zero)), min_valid_pts=int(min(nv)), p5_valid_pts=float(np.percentile(nv, 5)),
           rowdecile_valid_frac=(rowcov / 300).round(3).tolist(), frac_pixels_gt80_in_raw=float(np.mean(gt80)),
           resize_to=(rw, rh), crop_fraction_of_width=518 / rw, nearest_up_dup_factor=float(np.mean(np.array(n_after) / np.array(n_raw))), val_quantisation_step_m=1 / 256)
# nearest vs bilinear/cubic RGB sampling-centre offset (dest px) for 375->518 scale
s = 375 / 518; x = np.arange(518); nn = np.floor(x * s); ctr = (x + 0.5) * s - 0.5
res["nn_vs_centre_aligned_src_offset_px_mean_min_max(dest px)"] = [float(np.mean((ctr - nn) / s)), float(np.min((ctr - nn) / s)), float(np.max((ctr - nn) / s))]
# confirm cv2 nearest == floor(x*s) mapping
idx = np.tile(np.arange(1242, dtype=np.float32), (375, 1)); out = cv2.resize(idx, (rw, rh), interpolation=cv2.INTER_NEAREST)[0]
res["cv2_nearest_equals_floor_mapping"] = bool(np.array_equal(out, np.floor(np.arange(rw) * 1242 / rw)))
json.dump(res, open(f"{PARTS}/kitti_check_result.json", "w"), indent=1); print(json.dumps(res, indent=1))
