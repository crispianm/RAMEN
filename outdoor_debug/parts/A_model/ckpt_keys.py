"""CPU: load each checkpoint, build the model exactly as bench_joint.load_ours does, report key diffs.
Also reports the DPT head output_conv2 weights, and head deltas vs the released Metric-VDA-S."""
import sys, json, torch
REPO="/home/wg19671/.claude/jobs/a0d391cb/tmp/od"; sys.path.insert(0, REPO)
from models import build_model, trainable_modules
CK={"C3_seed45":"/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt",
"RAMEN_n46":"/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt",
"FT_n46":"/mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt"}
rel=torch.load("/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth",map_location="cpu",weights_only=False)
print("released metric-VDA-S keys:",len(rel))
for n,p in CK.items():
    ck=torch.load(p,map_location="cpu",weights_only=False)
    print("=====",n,"top-level keys:",list(ck.keys()))
    cfg=ck["cfg"]; m=cfg["model"]
    print(" cfg.model arch/metric/init/ckpt/freeze_depth_head/shared_temporal/vda_lora_rank:",m.get("arch"),m.get("metric"),m.get("vda_init"),m.get("vda_checkpoint_path"),m.get("freeze_depth_head"),m.get("shared_temporal"),m.get("vda_lora_rank"))
    print(" cfg.loss metric_depth_clamp:",cfg["loss"].get("metric_depth_clamp"),"data.max_depth",cfg["data"].get("max_depth"),"model has max_depth keys:",[k for k in m if "depth" in k.lower()])
    if m.get("shared_temporal"): cfg={**cfg,"model":{**m,"st_temp2_init":None}}
    net=build_model(cfg)
    mods=trainable_modules(net)
    md=torch.nn.ModuleDict(mods)
    for kk in ("trainable_ema","trainable"):
        if kk in ck:
            src=ck[kk]; dst=md.state_dict()
            miss=[k for k in dst if k not in src]; unexp=[k for k in src if k not in dst]
            shp=[k for k in src if k in dst and src[k].shape!=dst[k].shape]
            print(f" {kk}: n_src={len(src)} n_dst={len(dst)} missing={len(miss)} unexpected={len(unexp)} shape_mismatch={len(shp)}", miss[:5], unexp[:5])
    # frozen part: keys of full model NOT covered by trainable modules (come from released init)
    full=net.state_dict(); cov=set("%s.%s"%(a,k) for a,mod in mods.items() for k in mod.state_dict())
    print(" model.state_dict n=",len(full)," covered-by-ckpt modules; module names:",list(mods))
    for kk in ("trainable_ema","trainable"):
        if kk in ck: print("  ",kk,"key prefixes:",sorted({k.split('.')[0] for k in ck[kk]}))
    # load and compare head vs released
    net_sd=None
    md.load_state_dict(ck.get("trainable_ema",ck["trainable"]),strict=True)
    core=net.vda.core if hasattr(net,"vda") else net.core
    csd=core.state_dict()
    diffs={}
    for k,v in rel.items():
        if k in csd:
            d=(csd[k].float()-v.float()).abs().max().item(); diffs[k]=d
    chg=[k for k,d in diffs.items() if d>0]
    enc=[k for k in chg if k.startswith("pretrained")]
    print(f" vs released: {len(chg)}/{len(diffs)} tensors changed; encoder tensors changed: {len(enc)} (LoRA merged? lora_rank={m.get('vda_lora_rank')}); keys in core not in released: {[k for k in csd if k not in rel][:3]}; released not in core: {[k for k in rel if k not in csd][:3]}")
    for k in ("head.scratch.output_conv2.0.weight","head.scratch.output_conv2.2.weight","head.scratch.output_conv2.2.bias"):
        print("  ",k,"maxabs delta",diffs.get(k),"ckpt bias/abs mean", csd[k].float().abs().mean().item())
    print("   output_conv2.2.bias ckpt:",csd["head.scratch.output_conv2.2.bias"].tolist()," released:",rel["head.scratch.output_conv2.2.bias"].tolist())
