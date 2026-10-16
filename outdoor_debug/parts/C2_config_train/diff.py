import yaml
L=lambda n: yaml.safe_load(open(f'effective_{n}.yaml'))
A,B,C=L('c3_seed45'),L('ramen_n46'),L('ft_n46')
def flat(d,p=''):
    o={}
    if isinstance(d,dict):
        for k,v in d.items(): o.update(flat(v,f'{p}.{k}' if p else k))
    elif isinstance(d,list) and d and isinstance(d[0],dict):
        for i,v in enumerate(d):
            tag=v.get('type','')+('/'+v['root'].split('/')[-1] if 'root' in v else '')
            o.update(flat(v,f'{p}[{i}:{tag}]'))
    else: o[p]=d
    return o
fa,fb,fc=flat(A),flat(B),flat(C)
keys=sorted(set(fa)|set(fb)|set(fc))
print('| key | C3 s45 | RAMEN n46 | FT n46 |\n|---|---|---|---|')
for k in keys:
    v=[str(x.get(k,'<absent>')) for x in (fa,fb,fc)]
    if len(set(v))>1: print(f'| {k} | '+' | '.join(x if len(x)<90 else x[:87]+'...' for x in v)+' |')
