import torch, math
B='/mnt/DATA/DegradedDepth/checkpoints_v3/'
cks={'c3_s45':'eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt',
'ramen_n46':'eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt',
'ft_n46':'eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt'}
for n,p in cks.items():
    ck=torch.load(B+p,map_location='cpu',weights_only=False); o=ck['optimizer']
    stp=None
    for gi,g in enumerate(o['param_groups']):
        v=0.0;m2=0.0;cnt=0;steps=set(); zero=0
        for pi in g['params']:
            st=o['state'].get(pi)
            if st is None: zero+=1; continue
            s=float(st['step']); steps.add(s)
            bc=1-0.999**s
            v+=float((st['exp_avg_sq'].float()/bc).sum()); cnt+=st['exp_avg_sq'].numel()
            if float(st['exp_avg_sq'].float().max())==0: zero+=1
        print(f'{n} group{gi} lr0={g["initial_lr"]} nparams={len(g["params"])} numel={cnt} steps={sorted(steps)[:2]} no-state/zero-grad params={zero}  RMS-grad-norm(sqrt(sum v_hat))={math.sqrt(v):.4f}')
