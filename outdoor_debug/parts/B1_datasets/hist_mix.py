import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
import run_sources_lib as L  # = lines 1-31 of run_sources.py without the argv line
import torch, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
mass = dict(tartanair=310000, vkitti=40000, pointodyssey=99999, irs=100000, spring=4843, kitti_raw=299397)
silog = dict(tartanair=655, vkitti=655, pointodyssey=655, irs=None, spring=None, kitti_raw=None)  # silog_max_depth per C3 yaml (irs/kitti: absent)
msup = dict(tartanair=1, vkitti=1, pointodyssey=1, irs=1, spring=0, kitti_raw=1)
edges = np.linspace(-1, 4, 51); res = {}; H = {}
for name in mass:
    ds = L.make(name); rng = random.Random(9); vals = []; nsky = 0; ninv = 0; ntot = 0
    for c in rng.sample(range(len(ds)), 48):
        s = ds[c]; t = 16; m = s["mask"][0, t].numpy() > .5; d = s["depth"][0, t].numpy(); sk = s["sky"][0, t].numpy() > .5
        vals.append(d[m]); nsky += sk.sum(); ninv += (~m & ~sk).sum(); ntot += m.size
    v = np.concatenate(vals); n = len(v)
    h, _ = np.histogram(np.log10(np.clip(v, .1, 1e4)), bins=edges); H[name] = h / max(n, 1)
    cap = silog[name]
    res[name] = dict(frames=48, f_valid=n / ntot, f_sky=nsky / ntot, f_invalid=ninv / ntot, valid_gt80=float((v > 80).mean()), valid_gt200=float((v > 200).mean()), valid_gt655=float((v > 655).mean()),
                     pct=[float(x) for x in np.percentile(v, [1, 50, 99])], max=float(v.max()), silog_ceiling=cap, metric_sup=msup[name], mass=mass[name],
                     f_valid_pixels_in_SiLog=float((v < cap).mean()) if cap else 1.0)
# mixture: share of metric-supervised SiLog pixels (mass x f_valid x in-SiLog) by distance band
tot = sum(res[k]["mass"] * res[k]["f_valid"] * msup[k] for k in mass)
res["_mixture_note"] = "pixel share of the metric (SiLog-supervised) pixels by source and distance, weights = sampling_mass x valid frac (nyu/scannet++ omitted: indoor, not on disk)"
band = {}
for k in mass:
    if not msup[k]: continue
    w = res[k]["mass"] * res[k]["f_valid"]
    band[k] = dict(share_of_valid_pixels=w / tot)
res["_mixture"] = band
json.dump(res, open(f"{PARTS}/hist_mix_result.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k in mass:
    ax[0].plot(10 ** edges[:-1], H[k], label=f"{k} (mass {mass[k]//1000}k)")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("valid-pixel depth after preprocessing (m), 48 frames/source"); ax[0].set_ylabel("fraction of valid pixels per bin"); ax[0].legend(fontsize=7)
for x, c in ((80, "gray"), (655, "orange")): ax[0].axvline(x, color=c, ls="--")
ks = list(mass); ax[1].bar(ks, [res[k]["valid_gt80"] for k in ks], label=">80 m"); ax[1].bar(ks, [res[k]["valid_gt655"] for k in ks], label=">655 m"); ax[1].set_ylabel("fraction of valid pixels"); ax[1].legend(); ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(f"{OUT}/depth_distribution_across_datasets.png", dpi=120)
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in res.items() if not k.startswith("_")}, indent=0)); print(band)
