import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
from data.pointodyssey_dataset import PointOdysseyDataset, _PO_DEPTH_SCALE
import json
ds = PointOdysseyDataset(root=f"{T}/point_odyssey/train", keep_list=REPO+"/provenance/dn01_vdamix/po_filter_v1.json", keep_sha256="ff4851c6ad1d8bb0754ef9ef30aeb1c83562624041074acd41e41b5e9eedb34c", num_frames=32, image_size=(518,518), max_depth=999.0, seed=45, clip_stride=32, degrade=False)
rng = random.Random(4); scenes = {}
for i in rng.sample(range(len(ds)), 400):
    c = ds.clips[i]; f = c[1][c[3] + rng.randrange(32)]
    a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED); d = a.astype(np.float32) * _PO_DEPTH_SCALE
    scenes.setdefault(c[2], []).append((a.dtype.name, float((a == 0).mean()), float((a == 65535).mean()), float(d.max()), float(np.percentile(d[a > 0], 99)) if (a > 0).any() else 0.0))
tab = {k: [float(np.mean([x[j] for x in v])) for j in (1, 2, 3, 4)] for k, v in scenes.items()}
allz = [x[1] for v in scenes.values() for x in v]; allmax = [x[3] for v in scenes.values() for x in v]
res = dict(dtypes=sorted({x[0] for v in scenes.values() for x in v}), n_scenes_sampled=len(scenes), mean_zero_frac=float(np.mean(allz)), frac_frames_zero_gt5pct=float(np.mean(np.array(allz) > .05)), frames_with_65535=float(np.mean([x[2] > 0 for v in scenes.values() for x in v])),
           max_depth_m_over_sample=float(np.max(allmax)), p99_of_frame_max=float(np.percentile(allmax, 99)), scenes_with_zero_gt20pct=[k for k, v in tab.items() if v[0] > .2][:20])
# do zeros mark sky? ratio of zero pixels in the top third of the frame
tops = []
for i in rng.sample(range(len(ds)), 200):
    c = ds.clips[i]; a = cv2.imread(str(c[1][c[3]]), cv2.IMREAD_UNCHANGED); z = a == 0
    if z.mean() > .02: tops.append(float(z[: a.shape[0] // 3].sum() / z.sum()))
res["zero_pixels_in_top_third_share_(frames_with_>2%zero)"] = [len(tops), float(np.mean(tops)) if tops else None]
json.dump(res, open(f"{PARTS}/po_check_result.json", "w"), indent=1); print(json.dumps(res, indent=1))
