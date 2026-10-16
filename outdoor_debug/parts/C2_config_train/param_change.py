import torch, sys, re, collections, csv
rel=torch.load('/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth',map_location='cpu')
cks={'c3_s45':'eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt',
'ramen_n46':'eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt',
'ft_n46':'eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt'}
rows=[]
for n,p in cks.items():
    ck=torch.load('/mnt/DATA/DegradedDepth/checkpoints_v3/'+p,map_location='cpu',weights_only=False)
    for kind in ['trainable','trainable_ema']:
        tr=ck[kind]
        print(n,kind,len(tr),'EMA==live:',all(torch.equal(tr[k],ck['trainable'][k]) for k in tr) if kind=='trainable_ema' else '')
    tr=ck['trainable']
    # optimizer groups
    for gi,g in enumerate(ck['optimizer']['param_groups']):
        print(n,'group',gi,{k:v for k,v in g.items() if k!='params'},'nparams',len(g['params']))
    groups=collections.defaultdict(lambda:[0,0,0.0,0,[]])
    notin=[]
    for k,v in tr.items():
        if not v.is_floating_point(): continue
        top='.'.join(k.split('.')[:2]) if k.startswith(('head.','depth_head.')) else k.split('.')[0]
        # map to released key: head.X -> head.X
        rk=k
        if k.startswith('depth_head.'): rk='head.'+k[len('depth_head.'):]
        elif k.startswith('motion_modules.'): rk='head.'+k
        if rk in rel:
            d=(v.float()-rel[rk].float()); rn=rel[rk].float().norm().item()+1e-12
            groups[top][0]+=1; groups[top][1]+= (d.abs().max().item()==0); groups[top][2]+=d.norm().item()**2; groups[top][3]+=rn**2
            rows.append((n,rk,tuple(v.shape),d.abs().max().item(),d.norm().item()/rn))
        else: notin.append(k)
    print(n,'keys not in released:',len(notin),notin[:3])
    for top,(c,unch,dn,rn,_) in sorted(groups.items()):
        print(f'  {n} {top}: tensors={c} bitwise-unchanged={unch} rel_change={dn**.5/max(rn**.5,1e-12):.4f}')
with open('/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/C2_config_train/param_change_vs_released.csv','w') as f:
    w=csv.writer(f); w.writerow(['ckpt','key','shape','max_abs_diff','rel_l2_change']); w.writerows(rows)

print('--- output layers rel change')
import csv
for r in rows:
    if 'output_conv' in r[1] and r[0]!='x': print(r[0],r[1],r[2],'maxdiff=%.3g'%r[3],'rel=%.4f'%r[4])
