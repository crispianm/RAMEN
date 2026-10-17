#!/usr/bin/env python3
"""usage: class_shares.py <rows.csv> <batch_meta.json> <outdir>. Mass-reweighted pixel-class shares (pixels, pred-map L1, head-norm) under the TRUE mix vs the OLD local-6 mix (indoor NYU/ScanNet++ removed, same data)."""
import sys, json, pathlib, numpy as np, pandas as pd
rows, meta_p, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
df = pd.read_csv(rows); meta = json.load(open(meta_p)); mass = meta["sampling_mass"]
CL = ["near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky"]
MIX = {"true9": [s for s in mass if s != "spring"], "old_local6": ["tartanair", "vkitti", "pointodyssey_train", "pointodyssey_val", "irs", "kitti_raw"],
       "outdoor_only(ta,vk,po,kitti)": ["tartanair", "vkitti", "pointodyssey_train", "pointodyssey_val", "kitti_raw"]}
clip = df[df["clip"] >= 0]
tot_px = clip[(clip.group == "pix") & (clip.comp == "all") & (clip.metric == "total_px")].set_index(["ckpt", "batch", "clip"]).value
src_of = clip[["ckpt", "batch", "clip", "source"]].drop_duplicates().set_index(["ckpt", "batch", "clip"]).source
res = []
def rw(vals_by_src, srcs):
    """vals_by_src: Series indexed (ckpt,batch,clip) -> value; returns {ckpt: sum_s mass_s*mean_s over srcs}"""
    out_ = {}
    for ck in vals_by_src.index.get_level_values(0).unique():
        v = vals_by_src.loc[ck]; s = src_of.loc[ck].reindex(v.index)
        out_[ck] = sum(mass[x] * v[s == x].mean() for x in srcs if (s == x).any())
    return out_
def get(group, comp, metric):
    g = clip[(clip.group == group) & (clip.comp == comp) & (clip.metric == metric)].set_index(["ckpt", "batch", "clip"]).value
    return g
allidx = tot_px.index
def full_series(g):   # clips lacking the class -> 0
    return g.reindex(allidx).fillna(0.0)
for mix, srcs in MIX.items():
    pxs = {k: rw(full_series(get("pix", k, "px")), srcs) for k in CL}
    tp = rw(tot_px, srcs)
    for ck in tp:
        for k in CL:
            res.append(dict(mix=mix, ckpt=ck, comp="pixels", metric="frac_of_all_pixels", cls=k, value=pxs[k][ck] / tp[ck]))
        vsum = sum(pxs[k][ck] for k in CL if k != "sky")
        for k in CL:
            res.append(dict(mix=mix, ckpt=ck, comp="pixels", metric="frac_of_valid_pixels", cls=k, value=pxs[k][ck] / vsum if k != "sky" else np.nan))
    for comp in ("vda", "silog"):
        den = rw(full_series(get("clip_l1", comp, "pred_l1")), srcs)
        for k in CL:
            num = rw(full_series(get(f"cls_l1|{k}", comp, "pred_l1")), srcs)
            for ck in num:
                res.append(dict(mix=mix, ckpt=ck, comp=comp, metric="predmap_l1_share_of_clip_total", cls=k, value=num[ck] / den[ck] if den[ck] else np.nan))
                res.append(dict(mix=mix, ckpt=ck, comp=comp, metric="predmap_l1_per_valid_px_rel_to_near", cls=k, value=np.nan))
        for grp in ("output_conv", "all"):
            nums = {k: rw(full_series(get(f"cls|{k}|{grp}", comp, "l2_norm")), srcs) for k in CL}
            for ck in tp:
                s_ = sum(nums[k][ck] for k in CL)
                for k in CL:
                    res.append(dict(mix=mix, ckpt=ck, comp=comp, metric=f"{grp}_norm_share_of_summed_class_norms", cls=k, value=nums[k][ck] / s_ if s_ else np.nan))
    for grp in ("output_conv", "all"):
        stn = rw(full_series(get(f"clip|{grp}", "sky_term", "l2_norm")), srcs); fn = rw(full_series(get(f"clip|{grp}", "full", "l2_norm")), srcs); sn = rw(full_series(get(f"clip|{grp}", "silog", "l2_norm")), srcs)
        for ck in stn:
            res.append(dict(mix=mix, ckpt=ck, comp="sky_term", metric=f"{grp}_norm_over_full_norm", cls="-", value=stn[ck] / fn[ck]))
            res.append(dict(mix=mix, ckpt=ck, comp="sky_term", metric=f"{grp}_norm_over_silog_norm", cls="-", value=stn[ck] / sn[ck]))
R = pd.DataFrame(res); R = R[R.metric != "predmap_l1_per_valid_px_rel_to_near"]
R.to_csv(out / "class_shares_by_mix.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
for met in ["frac_of_all_pixels", "predmap_l1_share_of_clip_total", "output_conv_norm_share_of_summed_class_norms", "all_norm_share_of_summed_class_norms"]:
    t = R[R.metric == met]
    print(met); print(t.pivot_table(index=["comp", "cls"], columns=["mix", "ckpt"], values="value").round(4).to_string())
print(R[R.comp == "sky_term"].pivot_table(index="metric", columns=["mix", "ckpt"], values="value").round(3).to_string())
