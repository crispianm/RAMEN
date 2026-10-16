#!/usr/bin/env python3
"""Aggregate loss_diagnostics.csv (long format) -> summary tables (mean +- std over batches) + figure. usage: summarize.py <csv> <outdir_small> <outdir_big>"""
import sys, json, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, small, big = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
df = pd.read_csv(csv)
meta = json.load(open(pathlib.Path(csv).parent / "batch_meta.json"))
CK = ["released", "FT_n46", "RAMEN_n46", "C3_s45"]
CK = [c for c in CK if c in set(df.ckpt)]
nb = df.groupby("ckpt").batch.nunique().to_dict()
out = []


def ms(s):
    s = s.dropna()
    return pd.Series({"mean": s.mean(), "std": s.std(ddof=1) if len(s) > 1 else np.nan, "n": len(s)})


# generic per-batch table (everything except the per-clip 'source*' groups)
gen = df[~df.group.isin(["source", "source_pixels", "source_predgrad_l1"])]
summ = gen.groupby(["ckpt", "group", "item", "component", "metric"]).value.apply(ms).unstack().reset_index()
summ.to_csv(small / "summary_mean_std.csv", index=False)

# source table: pool clips. per batch sum of shares by source, then mean over ALL batches (absent source = 0 that batch)
src = df[df.group == "source"].copy()
src[["source", "grp"]] = src["item"].str.split("|", expand=True)
sp = df[df.group == "source_pixels"].copy()
rows = []
for ck in CK:
    nbat = nb[ck]
    spk = sp[sp.ckpt == ck]
    pivot = spk.groupby(["item", "metric"]).value.sum().unstack().fillna(0)
    tot_valid = pivot["valid_px"].sum(); tot_clips = pivot["n_clips"].sum(); tot_silog = pivot["silog_px"].sum()
    for sname in pivot.index:
        p = pivot.loc[sname]
        r = dict(ckpt=ck, source=sname, n_clips=int(p["n_clips"]), clip_share=p["n_clips"] / tot_clips, valid_px_share=p["valid_px"] / tot_valid,
                 silog_px_share=p["silog_px"] / tot_silog if tot_silog else np.nan, sky_px_per_clip=p["sky_px"] / p["n_clips"],
                 valid_frac_per_clip=p["valid_px"] / p["total_px"], sparse_flag=p["sparse_flag"] / p["n_clips"],
                 sampling_mass_share=meta["sampling_mass_local"].get(sname, np.nan) / sum(meta["sampling_mass_local"].values()))
        for grp in ("output_conv", "last_dpt", "all"):
            for comp in ("full", "vda", "silog"):
                s = src[(src.ckpt == ck) & (src.source == sname) & (src.grp == grp) & (src.component == comp)]
                for met, tag in (("norm_share_of_sum_of_clip_norms", "normshare"), ("projection_share_on_total", "projshare")):
                    ss = s[s.metric == met]
                    r[f"{grp}|{comp}|{tag}"] = ss.value.sum() / nbat           # expected share of a batch's gradient carried by this source
                r[f"{grp}|{comp}|clipnorm_mean"] = s[s.metric == "clip_grad_l2"].value.mean()
        pg = df[(df.group == "source_predgrad_l1") & (df.ckpt == ck) & (df.item == sname)]
        for comp in ("full", "vda", "silog"):
            r[f"predgrad_l1_per_clip|{comp}"] = pg[pg.component == comp].value.mean()
        rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(small / "summary_by_source.csv", index=False)

# class-share figure: share of the pred-map L1 gradient mass (per metre and per log-depth) by class, VDA / SiLog / full, per checkpoint
CLS = ["near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky", "invalid"]
col = {"near_lt10": "#2b6cb0", "mid_10_80": "#38a169", "far_80_655": "#d69e2e", "far_ge655_valid": "#dd6b20", "sky": "#805ad5", "invalid": "#a0aec0"}
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
for r_, metric in enumerate(["share_l1_per_m", "share_l1_per_log"]):
    for c_, comp in enumerate(["full", "vda", "silog"]):
        ax = axes[r_, c_]
        bottom = np.zeros(len(CK))
        for k in CLS:
            vals = []
            for ck in CK:
                s = gen[(gen.ckpt == ck) & (gen.group == "pred_grad") & (gen.item == k) & (gen.component == comp) & (gen.metric == metric)].value
                vals.append(s.mean() if len(s) else 0.0)
            vals = np.nan_to_num(np.array(vals))
            ax.bar(CK, vals, bottom=bottom, color=col[k], label=k if (r_, c_) == (0, 0) else None)
            bottom += vals
        ax.set_title(f"{comp}: {'per metre' if metric.endswith('_m') else 'per log-depth (|g|*pred)'}", fontsize=10)
        ax.set_ylabel("share of pred-map L1 gradient mass")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=15)
fig.legend(loc="lower center", ncol=6, fontsize=9)
fig.suptitle(f"Gradient w.r.t. predicted depth by GT pixel class (mean over {min(nb.values())}-{max(nb.values())} real train batches of 4 clips x 8 frames)", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(big / "class_gradient_share.png", dpi=130)
fig.savefig(small / "class_gradient_share.png", dpi=70)
print("ok", small)
