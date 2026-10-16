import numpy as np, json, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
B="/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/D_baseline/"
M=["mvda_s_stock","ramen_s45","ramen_n46","ft_n46"]; LAB={"mvda_s_stock":"released Metric-VDA-S","ramen_s45":"RAMEN C3 s45","ramen_n46":"RAMEN n46","ft_n46":"FT n46"}
# 1) six-clip side-by-sides
v=np.load(B+"vis_frames.npz"); keys=sorted({k.rsplit("|",1)[0] for k in v.files})
for k in keys:
    ds=k.split("|")[0]; vmax={"kitti":80,"vkitti":200}.get(ds,10)
    gt=v[k+"|gt"]; fig,ax=plt.subplots(1,8,figsize=(32,3.2))
    ax[0].imshow(v[k+"|rgb"]); ax[0].set_title("RGB")
    g=np.where(gt>0,gt,np.nan); ax[1].imshow(g,cmap="turbo",vmin=0,vmax=vmax); ax[1].set_title(f"GT (0-{vmax} m)")
    for i,m in enumerate(M):
        ax[2+i].imshow(v[k+"|"+m],cmap="turbo",vmin=0,vmax=vmax); ax[2+i].set_title(f"{LAB[m]} (med {np.median(v[k+'|'+m]):.1f} m)")
    ax[6].imshow(v[k+"|sky"],cmap="gray"); ax[6].set_title("sky mask"); ax[7].axis("off")
    ax[7].text(0,.5,"shared colour scale\n0-%d m (turbo), mid frame"%vmax,fontsize=12)
    for a in ax[:7]: a.axis("off")
    plt.tight_layout(); plt.savefig(B+"sbs_%s.png"%k.replace("|","_"),dpi=60); plt.close()
# 2) pred vs GT
s=np.load(B+"samples.npz"); bins=np.geomspace(1,700,28)
fig,axs=plt.subplots(1,2,figsize=(13,5.5))
for a,ds in zip(axs,["kitti","vkitti"]):
    a.axvspan(80,700,color="#fde0dd",alpha=.6,label=">80 m"); a.plot([1,700],[1,700],"k--",lw=1)
    for m,c in zip(M,["C3","C0","C2","C1"]):
        g=s[f"{m}|{ds}|gt"]; p=s[f"{m}|{ds}|pred"]; idx=np.digitize(g,bins); x=[];q=[]
        for b in range(1,len(bins)):
            sel=p[idx==b]
            if len(sel)>=30: x.append(np.sqrt(bins[b-1]*bins[b])); q.append(np.percentile(sel,[25,50,75]))
        q=np.array(q); a.plot(x,q[:,1],c=c,label=LAB[m]); a.fill_between(x,q[:,0],q[:,2],color=c,alpha=.12)
    a.set_xscale("log");a.set_yscale("log");a.set_xlabel("GT depth (m)");a.set_ylabel("predicted depth (m), raw");a.set_title(ds+" DEV (binned median, IQR)");a.legend(fontsize=8)
plt.tight_layout();plt.savefig(B+"pred_vs_gt_outdoor.png",dpi=80)
# 3) KITTI 0023 frame 109
for cell in ("llb3","b0s0"):
    c=np.load(B+"case_kitti0023_f109.npz"); sky=c["sky"].astype(np.uint8)
    for scale in ("linear","log"):
        fig,ax=plt.subplots(1,5,figsize=(25,3.4))
        ax[0].imshow(np.clip(c[cell+"|rgb"],0,1)); ax[0].set_title(f"input ({cell})")
        norm=Normalize(0,700) if scale=="linear" else LogNorm(5,700)
        for i,m in enumerate(M):
            d=c[cell+"|"+m]; im=ax[1+i].imshow(d,cmap="turbo",norm=norm); ax[1+i].contour(sky,[0.5],colors="w",linewidths=0.8)
            ax[1+i].set_title(f"{LAB[m]}: sky med {np.median(d[sky>0]):.0f} m")
        for a in ax: a.axis("off")
        fig.colorbar(im,ax=ax[-1],fraction=.05); plt.savefig(B+f"kitti0023_f109_{cell}_{scale}.png",dpi=60,bbox_inches="tight"); plt.close()
# 4) per-frame sky median distribution
pf=json.load(open("kitti_sky_perframe.json"))
fig,axs=plt.subplots(1,2,figsize=(12,4))
for a,cell in zip(axs,("b0s0","llb3")):
    a.boxplot([[r[2] for r in pf[f"{cell}|{m}"]] for m in M],labels=[LAB[m].replace(" ","\n") for m in M],showfliers=False)
    a.set_yscale("log");a.set_ylabel("per-frame median predicted depth in sky mask (m)");a.set_title(f"KITTI DEV (all frames with sky), cell {cell}")
plt.tight_layout();plt.savefig(B+"kitti_sky_perframe_box.png",dpi=80)
# 5) horse
h=np.load(B+"case_horse.npz"); H=h["rgb25"].shape[0]; sb=int(.15*H)
fig,ax=plt.subplots(1,5,figsize=(30,3.8)); ax[0].imshow(h["rgb25"]); ax[0].axhline(sb,color="w"); ax[0].set_title("horse frame 25 (sky band = top 15%)")
for i,m in enumerate(M):
    d=h[m+"_fp32"][25]; im=ax[1+i].imshow(d,cmap="turbo",norm=LogNorm(4.05,641.81)); ax[1+i].axhline(sb,color="w",lw=1.5)
    ax[1+i].set_title(f"{LAB[m]}: sky-band med {np.median(d[:sb]):.0f} m")
for a in ax: a.axis("off")
fig.colorbar(im,ax=ax[-1],fraction=.05); plt.savefig(B+"horse_f25_log.png",dpi=55,bbox_inches="tight")
