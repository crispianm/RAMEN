"""CPU synthetic tests: representability of 1,5,10,30,80,200,600 m by the metric head's output stack and by the alternative DA-V2 sigmoid x80 head;
dead-gradient / clamp behaviour of the training losses; reciprocal of zeros; interpolation scale; bf16 quantisation. Run: python unit_tests.py"""
import sys, math, torch, torch.nn as nn, torch.nn.functional as F
REPO="/home/wg19671/.claude/jobs/a0d391cb/tmp/od"; sys.path.insert(0,REPO)
DEPTHS=[1,5,10,30,80,200,600]
print("== T1 Metric-VDA head tail: Conv1x1 -> ReLU (DPT output_conv2[3]) -> F.relu (VideoDepthAnything.forward) ==")
tail=nn.Sequential(nn.ReLU(),); 
for d in DEPTHS:
    z=torch.tensor(float(d)); o=F.relu(tail(z)); print(f" depth {d:>4} m: pre-act z={d} -> out {o.item():.4f} (representable: {abs(o.item()-d)<1e-6})")
print(" negative pre-acts -> 0 exactly, gradient 0:", end=" ")
z=torch.tensor([-1.0,0.0,1.0],requires_grad=True); F.relu(F.relu(z)).sum().backward(); print(z.grad.tolist(),"(z=0 has 0 grad too)")
print("== T2 bf16 quantisation of the head output (VideoDepthAnything head casts `out.to(ori_type)`; ori_type=bf16 under train.py autocast) ==")
for d in DEPTHS:
    v=d*1.0123; q=torch.tensor(float(v)).bfloat16().float().item(); print(f" {v:>8.3f} m -> bf16 {q:.4f}  rel.err {(q-v)/v:+.4%}  (spacing at this magnitude {2**(math.floor(math.log2(d))-7):.4f} m)")
print("== T3 DA-V2 sigmoid x max_depth head (dav2_net head='sigmoid', max_depth 80) ==")
for md in (80.0,):
    for d in DEPTHS:
        if d<md:
            z=math.log(d/(md-d)); s=torch.tensor(z,requires_grad=True); (md*torch.sigmoid(s)).backward(); print(f" {d:>4} m: z={z:+.3f} ok, d(out)/dz={s.grad.item():.4f}")
        else: print(f" {d:>4} m: NOT representable (output < {md} m; fp32 sigmoid saturates to exactly {md} m at z>=~17, with ~0 gradient)")
z=torch.tensor(17.0,requires_grad=True); o=80*torch.sigmoid(z); o.backward(); print(f" saturation check: 80*sigmoid(17)={o.item()!r}, grad={z.grad.item():.3e}")
print("== T4 training-loss gradient of predicted depth (train.py _vda_disp with metric_depth_clamp [0.1,1e5] and the SiLog clamp) ==")
import train
batch={"depth":torch.full((1,1,1,1,1),30.0),"mask":torch.ones(1,1,1,1,1)}
for pdv in [0.0,0.05,0.1,1.0,30.0,1e3,1e5,2e5]:
    p=torch.full((1,1,1,1,1),pdv,requires_grad=True)
    pd,gd,m=train._vda_disp(p,batch,False,(0.1,1e5))
    (pd).sum().backward(); g1=p.grad.item(); p.grad=None
    pd_d=p.reshape(1).float().clamp(0.1,1e5); ((torch.log(torch.tensor(30.0))-torch.log(pd_d))**2).sum().backward(); g2=p.grad.item()
    print(f" pred={pdv:>9}: pd=1/clamp -> {pd.item():.6g}, d pd/d pred={g1:+.3e}, d(logerr^2)/d pred={g2:+.3e}")
print(" -> pred <0.1 m or >1e5 m: ZERO gradient from the SSI-disparity and SiLog losses (and ReLU zero-grad for z<=0).")
print("== T5 reciprocal handling ==")
z=torch.tensor([0.0,1e-6,1e-3,0.5]); 
print(" train._vda_disp floor 1e-3 (no clamp cfg):", (1/z.clamp_min(1e-3)).tolist(), "| with metric_depth_clamp(0.1,1e5):", (1/z.clamp(0.1,1e5)).tolist())
print(" train.evaluate / bench_joint.score_clip floor 1e-3: pred 0 m -> disparity", (1/torch.tensor(0.0).clamp(min=1e-3)).item(), "(vs a 30 m pixel: %.4f)"%(1/30))
print(" GT sky 655 m -> disparity %.5f; 1e5 m -> %.1e; train gt disparity floor 1e-3 m => max disparity 1000"%(1/655,1/1e5))
print("== T6 resize scale: bilinear align_corners=True on constant/step fields ==")
c=torch.full((1,1,10,10),600.); print(" const 600 -> unique", F.interpolate(c,size=(37,53),mode="bilinear",align_corners=True).unique().tolist())
s=torch.cat([torch.full((1,1,10,5),2.),torch.full((1,1,10,5),600.)],3); u=F.interpolate(s,size=(10,50),mode="bilinear",align_corners=True); print(" step 2|600 m upsampled: values at the edge:", [round(v,1) for v in u[0,0,0,20:30].tolist()], "(flying-pixel depths between the two; metric scale of each plateau unchanged)")
