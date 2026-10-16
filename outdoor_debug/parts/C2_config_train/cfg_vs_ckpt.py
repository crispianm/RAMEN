import torch, yaml
B='/mnt/DATA/DegradedDepth/checkpoints_v3/'
pairs={'c3_seed45':('eval_20261009/RAMEN_train_dn01_v3_bfix_C3_B/runs/dn01_v3_bfix_C3_B/step12000.pt'),
'ramen_n46':'eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt',
'ft_n46':'eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt'}
def flat(d,p=''):
    o={}
    if isinstance(d,dict):
        for k,v in d.items(): o.update(flat(v,f'{p}.{k}'))
    else: o[p]=d
    return o
for n,p in pairs.items():
    ck=torch.load(B+p,map_location='cpu',weights_only=False)['cfg']
    eff=yaml.safe_load(open(f'effective_{n}.yaml'))
    a,b=flat(ck),flat(eff); diffs=[k for k in sorted(set(a)|set(b)) if a.get(k,'<absent>')!=b.get(k,'<absent>')]
    print(n,'checkpoint-embedded cfg vs repo-resolved effective cfg: differing keys =',len(diffs)); 
    for k in diffs[:12]: print('   ',k,'ckpt=',str(a.get(k,'<absent>'))[:80],'| repo=',str(b.get(k,'<absent>'))[:80])
