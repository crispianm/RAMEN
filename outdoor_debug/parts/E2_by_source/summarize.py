#!/usr/bin/env python3
"""usage: summarize.py <grad_by_source_rows.csv> <batch_meta.json> <outdir>   -> summary CSVs in outdir"""
import sys, json, pathlib
import numpy as np, pandas as pd

csv, meta_p, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
df = pd.read_csv(csv)
meta = json.load(open(meta_p))
mass = meta["sampling_mass"]; tm = sum(mass.values())
SRC = meta["sources"]
GROUPS = {"pointodyssey": ["pointodyssey_train", "pointodyssey_val"], "indoor(irs+nyu+scannetpp)": ["irs", "nyu_train", "scannetpp_train"]}
rng = np.random.default_rng(0)
CKS = [c for c in ["released", "FT_n46", "RAMEN_n46"] if c in set(df.ckpt)]


def piv(d, group, comp, metric):
    s = d[(d.group == group) & (d.comp == comp) & (d.metric == metric)]
    return s.set_index(["batch", "clip"]).value


def share_table(d, getval, nb):
    """getval: DataFrame index (batch, clip) -> value, plus source. returns pooled share per source + bootstrap CI over batches"""
    t = getval.reset_index().merge(d[["batch", "clip", "source"]].drop_duplicates(), on=["batch", "clip"])
    byb = t.pivot_table(index="batch", columns="source", values="value", aggfunc="sum").reindex(columns=SRC).fillna(0.0)
    tot = byb.sum(axis=1)
    pooled = byb.sum() / byb.sum().sum()
    perb = byb.div(tot, axis=0).mean()
    bs = []
    arr = byb.values
    for _ in range(1000):
        ix = rng.integers(0, len(arr), len(arr)); a = arr[ix].sum(0); bs.append(a / a.sum())
    bs = np.array(bs)
    return pooled, perb, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0), byb


rows = []
cls_rows = []
for ck in CKS:
    d = df[df.ckpt == ck]
    nb = d.batch.nunique()
    clip = d[d["clip"] >= 0]
    pix = clip[clip.group == "pix"]
    ncl = pix[pix.comp == "all"].drop_duplicates(["batch", "clip"]).groupby("source").size().reindex(SRC).fillna(0)
    vpx = pix[(pix.comp == "all") & (pix.metric == "valid_px")].groupby("source").value.sum().reindex(SRC).fillna(0)
    spx = pix[(pix.comp == "all") & (pix.metric == "silog_px")].groupby("source").value.sum().reindex(SRC).fillna(0)
    tpx = pix[(pix.comp == "all") & (pix.metric == "total_px")].groupby("source").value.sum().reindex(SRC).fillna(0)
    clspx = {k: pix[(pix.comp == k) & (pix.metric == "px")].groupby("source").value.sum().reindex(SRC).fillna(0) for k in ["near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky"]}
    base = pd.DataFrame({"n_clips": ncl, "clip_share": ncl / ncl.sum(), "mass_share": pd.Series({s: mass[s] / tm for s in SRC}),
                         "valid_px_share": vpx / vpx.sum(), "silog_px_share": spx / spx.sum(), "valid_frac_of_pixels": vpx / tpx})
    for k, v in clspx.items(): base[f"px_share_{k}"] = v / v.sum() if v.sum() else np.nan
    for grp in ("output_conv", "all"):
        for comp in ("full", "vda", "silog", "sky_term"):
            gv = clip[(clip.group == f"clip|{grp}") & (clip.comp == comp) & (clip.metric == "l2_norm")].set_index(["batch", "clip"]).value.to_frame("value")
            pooled, perb, lo, hi, byb = share_table(clip, gv["value"], nb)
            base[f"{grp}|{comp}|norm_share_pooled"] = pooled
            base[f"{grp}|{comp}|norm_share_perbatch_mean"] = perb
            base[f"{grp}|{comp}|norm_share_ci_lo"] = lo; base[f"{grp}|{comp}|norm_share_ci_hi"] = hi
            e = (gv["value"] ** 2).to_frame("value")
            base[f"{grp}|{comp}|energy_share_pooled"] = share_table(clip, e["value"], nb)[0]
            # projection onto batch total: sum_dot / sum_total_norm^2 (per batch), mean over batches
            dt = clip[(clip.group == f"clip|{grp}") & (clip.comp == comp) & (clip.metric == "dot_with_total")].merge(
                d[(d["clip"] == -1) & (d.group == f"total|{grp}") & (d.comp == comp)][["batch", "value"]].rename(columns={"value": "tot"}), on="batch")
            dt["proj"] = dt.value / dt.tot ** 2
            base[f"{grp}|{comp}|projection_share_mean_over_batches"] = dt.groupby("source").proj.sum().reindex(SRC).fillna(0) / nb
            base[f"{grp}|{comp}|clip_norm_mean"] = gv.reset_index().merge(clip[["batch", "clip", "source"]].drop_duplicates(), on=["batch", "clip"]).groupby("source").value.mean().reindex(SRC)
    for comp in ("full", "vda", "silog"):
        l1 = clip[(clip.group == "clip_l1") & (clip.comp == comp)].groupby("source").value.sum().reindex(SRC).fillna(0)
        base[f"predmap_l1_share|{comp}"] = l1 / l1.sum()
    base.insert(0, "ckpt", ck)
    rows.append(base.reset_index().rename(columns={"index": "source"}))
    # source x class x loss shares (class-restricted gradient; share of the class's summed clip norms)
    for grp in ("output_conv", "all"):
        for comp in ("vda", "silog"):
            for k in ["near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky"]:
                g = clip[(clip.group == f"cls|{k}|{grp}") & (clip.comp == comp) & (clip.metric == "l2_norm")]
                s = g.groupby("source").value.sum().reindex(SRC).fillna(0)
                tot = s.sum()
                for sname in SRC:
                    cls_rows.append(dict(ckpt=ck, group=grp, comp=comp, cls=k, source=sname, norm_sum=s[sname], share=s[sname] / tot if tot else np.nan,
                                         px_share=base.loc[sname, f"px_share_{k}"], mass_share=base.loc[sname, "mass_share"]))
S = pd.concat(rows)
S.to_csv(out / "summary_by_source.csv", index=False)
pd.DataFrame(cls_rows).to_csv(out / "summary_by_source_class.csv", index=False)

# aggregated groups (PO train+val; indoor)
agg = []
for ck in CKS:
    s = S[S.ckpt == ck].set_index("source")
    numeric = [c for c in s.columns if c not in ("ckpt",) and not c.endswith("clip_norm_mean") and "ci_" not in c]
    for gname, members in list(GROUPS.items()) + [("kitti_raw", ["kitti_raw"]), ("tartanair", ["tartanair"]), ("vkitti", ["vkitti"]), ("spring", ["spring"]), ("irs", ["irs"]), ("nyu_train", ["nyu_train"]), ("scannetpp_train", ["scannetpp_train"]), ("outdoor-ish(kitti+vkitti)", ["kitti_raw", "vkitti"])]:
        r = {"ckpt": ck, "group": gname}
        for c in numeric:
            r[c] = s.loc[members, c].sum() if c != "n_clips" else s.loc[members, c].sum()
        agg.append(r)
pd.DataFrame(agg).to_csv(out / "summary_grouped.csv", index=False)
print("ok")

# ---- mass-reweighted expectation: share_s = mass_s * E[x | clip from s] / sum_s' mass_s' * E[x | s']   (removes the 80-clip sampling noise of source counts; spring has no clip in 20 batches -> omitted, 0.4% mass)
mw = []
for ck in CKS:
    d = df[df.ckpt == ck]; clip = d[d["clip"] >= 0]
    nb_ = d.batch.nunique()
    def per_clip(group, comp, metric):
        return clip[(clip.group == group) & (clip.comp == comp) & (clip.metric == metric)][["batch", "clip", "source", "value"]]
    quantities = {"clips": None}
    qs = {}
    for grp in ("output_conv", "all"):
        for comp in ("full", "vda", "silog", "sky_term"):
            qs[f"{grp}|{comp}|norm"] = per_clip(f"clip|{grp}", comp, "l2_norm")
    qs["valid_px"] = per_clip("pix", "all", "valid_px"); qs["silog_px"] = per_clip("pix", "all", "silog_px")
    for k in ["near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky"]:
        qs[f"px_{k}"] = per_clip("pix", k, "px")
    for k in ["near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky"]:
        for comp in ("vda", "silog"):
            g = clip[(clip.group == f"cls|{k}|output_conv") & (clip.comp == comp) & (clip.metric == "l2_norm")][["batch", "clip", "source", "value"]]
            # clips without that class contribute 0 norm
            allc = clip[(clip.group == "pix") & (clip.comp == "all") & (clip.metric == "total_px")][["batch", "clip", "source"]].merge(g, on=["batch", "clip", "source"], how="left").fillna({"value": 0.0})
            qs[f"output_conv|{comp}|{k}|norm"] = allc
    for qn, q in qs.items():
        srcs = [s for s in SRC if (q.source == s).any()]
        means = {s: q[q.source == s].value.mean() for s in srcs}
        w = {s: mass[s] * means[s] for s in srcs}; tw = sum(w.values())
        # bootstrap clips within source
        bs = []
        vals = {s: q[q.source == s].value.values for s in srcs}
        for _ in range(500):
            ww = {s: mass[s] * rng.choice(vals[s], len(vals[s])).mean() for s in srcs}; t = sum(ww.values())
            bs.append([ww[s] / t if t else np.nan for s in srcs])
        bs = np.array(bs)
        for j, s in enumerate(srcs):
            mw.append(dict(ckpt=ck, quantity=qn, source=s, n_clips=len(vals[s]), mass_share=mass[s] / tm, mean_per_clip=means[s], reweighted_share=w[s] / tw if tw else np.nan,
                           ci_lo=np.nanpercentile(bs[:, j], 2.5), ci_hi=np.nanpercentile(bs[:, j], 97.5)))
MW = pd.DataFrame(mw)
MW.to_csv(out / "summary_massweighted.csv", index=False)
g_ = []
for (ck, qn), t in MW.groupby(["ckpt", "quantity"]):
    t = t.set_index("source")
    for gname, mem in [("indoor(irs+nyu+scannetpp)", ["irs", "nyu_train", "scannetpp_train"]), ("kitti_raw", ["kitti_raw"]), ("tartanair", ["tartanair"]), ("pointodyssey", ["pointodyssey_train", "pointodyssey_val"]), ("vkitti", ["vkitti"]), ("irs", ["irs"]), ("nyu_train", ["nyu_train"]), ("scannetpp_train", ["scannetpp_train"])]:
        mem = [m_ for m_ in mem if m_ in t.index]
        g_.append(dict(ckpt=ck, quantity=qn, group=gname, mass_share=t.loc[mem, "mass_share"].sum(), reweighted_share=t.loc[mem, "reweighted_share"].sum()))
pd.DataFrame(g_).to_csv(out / "summary_massweighted_grouped.csv", index=False)
print("ok mw")
