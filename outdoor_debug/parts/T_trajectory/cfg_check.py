import torch, yaml, sys
B='/mnt/DATA/DegradedDepth/checkpoints_v3/'
P={'ft_n46':{6000:'outdoor_debug_20261016/ft_n46/step6000.pt',8000:'outdoor_debug_20261016/ft_n46/step8000.pt',12000:'eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt'},
 'c3_n46':{6000:'outdoor_debug_20261016/c3_n46/step6000.pt',8000:'outdoor_debug_20261016/c3_n46/step8000.pt',12000:'eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt'}}
EFF={'ft_n46':'effective_ft_n46.yaml','c3_n46':'effective_ramen_n46.yaml'}
def flat(d,p=''):
    o={}
    if isinstance(d,dict):
        for k,v in d.items(): o.update(flat(v,f'{p}.{k}'))
    else: o[p]=d
    return o
R='/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/C2_config_train/'
for m,st in P.items():
    eff=flat(yaml.safe_load(open(R+EFF[m])))
    cf={s:flat(torch.load(B+p,map_location='cpu',weights_only=False)['cfg']) for s,p in st.items()}
    for s,a in cf.items():
        d=[k for k in sorted(set(a)|set(eff)) if a.get(k,'<absent>')!=eff.get(k,'<absent>')]
        print(m,s,'vs effective yaml: differing keys',len(d))
        for k in d[:15]: print('    ',k,'ckpt=',str(a.get(k,'<absent>'))[:70],'| eff=',str(eff.get(k,'<absent>'))[:70])
    for s in (6000,8000):
        d=[k for k in sorted(set(cf[s])|set(cf[12000])) if cf[s].get(k,'<a>')!=cf[12000].get(k,'<a>')]
        print(m,s,'vs step12000 embedded cfg: differing keys',len(d), d[:10])
