"""CPU real-frame probe on DEV VKITTI (Scene01/clone Camera_0, train data; sky GT = 655.35 m) and KITTI-raw train drive (no dense GT).
Per checkpoint: output range, fractions at the head's floor (exact 0 / <0.1 m) and near the observed max, fractions >= 80/100/655/1e5 m,
median sky / horizon-ish depth, fp32-vs-bf16 autocast difference, final-conv capacity from the real feature maxima. Usage: real_frames.py OUT.json"""
import sys, json, numpy as np, torch, cv2, torch.nn.functional as F
REPO="/home/wg19671/.claude/jobs/a0d391cb/tmp/od"; sys.path.insert(0, REPO)
torch.set_num_threads(16)
from models import build_model, trainable_modules
D="/mnt/DATA/DegradedDepth/data/TRAIN"
CK={"released_MetricVDAS":None,
"C3_seed45":"/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt",
"RAMEN_n46":"/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt",
"FT_n46":"/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt"}
H,W,T=280,924,8
def vkitti(first):
    fr,gt=[],[]
    for i in range(first,first+T):
        im=cv2.cvtColor(cv2.imread(f"{D}/vkitti/Scene01/clone/frames/rgb/Camera_0/rgb_{i:05d}.jpg"),cv2.COLOR_BGR2RGB)
        d=cv2.imread(f"{D}/vkitti/Scene01/clone/frames/depth/Camera_0/depth_{i:05d}.png",cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.0
        fr.append(cv2.resize(im,(W,H),interpolation=cv2.INTER_CUBIC)); gt.append(cv2.resize(d,(W,H),interpolation=cv2.INTER_NEAREST))
    x=torch.from_numpy(np.stack(fr)).permute(3,0,1,2)[None].float()/255.
    return x,np.stack(gt)
def kitti(drive="2011_09_26_drive_0009_sync",first=0):
    fr=[]
    for i in range(first,first+T):
        im=cv2.cvtColor(cv2.imread(f"{D}/kitti_raw/2011_09_26/{drive}/image_02/data/{i:010d}.png"),cv2.COLOR_BGR2RGB)
        fr.append(cv2.resize(im,(W,H),interpolation=cv2.INTER_CUBIC))
    return torch.from_numpy(np.stack(fr)).permute(3,0,1,2)[None].float()/255., None
def build(p):
    if p is None:
        from models.vda_net import VDANet
        return VDANet(encoder="vits",metric=True,init="released",checkpoint_path="/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth").eval()
    ck=torch.load(p,map_location="cpu",weights_only=False); cfg=ck["cfg"]
    if cfg["model"].get("shared_temporal"): cfg={**cfg,"model":{**cfg["model"],"st_temp2_init":None}}
    net=build_model(cfg); torch.nn.ModuleDict(trainable_modules(net)).load_state_dict(ck["trainable_ema"],strict=True); return net.eval()
def fwd(net,x):
    with torch.no_grad():
        if hasattr(net,"depth_and_feats"): return net.depth_and_feats(x)[0][0,0]
        return net(x)[0,0]
def stats(d,gt,tag):
    d=d.float().numpy(); r={"tag":tag,"min":float(d.min()),"max":float(d.max()),"median":float(np.median(d)),"p99":float(np.percentile(d,99)),
      "frac_exact0":float((d==0).mean()),"frac_lt0.1m":float((d<0.1).mean()),"frac_le_1pct_of_floor0.0_(<=1e-2m)":float((d<=1e-2).mean()),
      "frac_ge_0.99max":float((d>=0.99*d.max()).mean()),"frac_ge80":float((d>=80).mean()),"frac_ge100":float((d>=100).mean()),"frac_ge655":float((d>=655).mean()),"frac_ge1e5":float((d>=1e5).mean())}
    if gt is not None:
        sky=gt>=655; valid=(gt>0)&(gt<655)
        r["gt_sky_frac"]=float(sky.mean())
        if sky.any(): r["pred_sky_median"]=float(np.median(d[sky])); r["pred_sky_p10_p90"]=[float(np.percentile(d[sky],10)),float(np.percentile(d[sky],90))]; r["frac_sky_exact0"]=float((d[sky]==0).mean())
        for lo,hi in [(0,10),(10,30),(30,80),(80,655)]:
            m=(gt>lo)&(gt<=hi)&valid
            if m.sum()>50: r[f"gt_{lo}-{hi}m"]={"n":int(m.sum()),"pred_median":float(np.median(d[m])),"gt_median":float(np.median(gt[m])),"absrel_raw":float((np.abs(d[m]-gt[m])/gt[m]).mean()),"med_ratio_pred/gt":float(np.median(d[m]/gt[m]))}
    return r
out={}
scenes={"vkitti_s01_clone_f0":vkitti(0),"vkitti_s01_clone_f300":vkitti(300),"kitti_raw_0009_f0":kitti()}
for n,p in CK.items():
    net=build(p); out[n]={}
    core=net.vda.core if hasattr(net,"vda") else net.core
    feat={}
    h=core.head.scratch.output_conv2[2].register_forward_hook(lambda m,i,o:feat.__setitem__("h",i[0].detach()))
    for sn,(x,gt) in scenes.items():
        d=fwd(net,x); r=stats(d,gt,sn)
        # input features of the final 1x1 conv and capacity
        hh=feat["h"]; w=core.head.scratch.output_conv2[2].weight.detach().flatten(); b=core.head.scratch.output_conv2[2].bias.item()
        r["final_conv_in_feat_max"]=float(hh.max()); r["final_conv_w_pos_sum"]=float(w.clamp(min=0).sum()); r["final_conv_bias"]=b
        r["max_preact_observed"]=float(((hh*w.view(1,-1,1,1)).sum(1)+b).max())  # pre-ReLU at head res (before bilinear to input res if different)
        with torch.autocast("cpu",dtype=torch.bfloat16): db=fwd(net,x)
        r["bf16_vs_fp32_rel_diff_median"]=float(((db.float()-d).abs()/d.clamp(min=1e-3)).median()); r["bf16_out_dtype"]=str(db.dtype)
        out[n][sn]=r; print(n,sn,{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items() if not isinstance(v,dict)},flush=True)
        for k,v in r.items():
            if isinstance(v,dict): print("   ",k,v)
        if n in("FT_n46","released_MetricVDAS","RAMEN_n46") and sn=="vkitti_s01_clone_f300":
            np.save(f"/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/A_model/{n}_{sn}_depth.npy",d[3].numpy()) if __import__("os").path.isdir("/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/A_model") else None
    h.remove(); del net
json.dump(out,open(sys.argv[1],"w"),indent=1)
