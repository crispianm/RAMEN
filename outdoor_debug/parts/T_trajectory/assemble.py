import csv, numpy as np, collections
H='.'
out=[]
for r in csv.DictReader(open('trajectory_eval_part.csv')): out.append((r['model'],r['step'],r['dataset'],r['quantity'],r['value'],r['n']))
acc=collections.defaultdict(list)
for r in csv.DictReader(open('loss_diagnostics_traj.csv')):
    ck=r['ckpt']; m,s=ck.rsplit('_',1); m=m if m!='released' else 'released'
    if ck=='released_0': m,s='released','0'
    g,i,c,me,v=r['group'],r['item'],r['component'],r['metric'],float(r['value'])
    q=None
    if g=='param_grad' and i in('output_conv','all') and me=='l2_norm' and c in('vda','silog','full','sky_term'): q=f'{c}_l2_{i}'
    elif g=='param_cos' and i=='output_conv' and c=='vda_vs_silog': q='cos_vda_silog_output_conv' if me=='cosine' else 'silog_over_vda_norm_output_conv'
    elif g=='pred_grad' and c=='full' and me=='share_l1_per_m' and i in('near_lt10','mid_10_80','far_80_655','far_ge655_valid','sky'): q=f'full_L1share_{i}'
    elif g=='pred_grad' and c=='full' and me=='l1_per_px_per_m' and i in('near_lt10','mid_10_80','far_80_655','sky'): q=f'full_L1_per_px_{i}'
    elif g=='pred_stat' and me in('median_pred_over_gt','median_pred_m','frac_pred_lt_80m'): q=f'{me}_{i}'
    elif g=='loss_value' and c in('full','vda','silog','sky_term'): q=f'loss_{c}'
    if q and np.isfinite(v): acc[(m,s,q)].append(v)
for (m,s,q),v in sorted(acc.items()):
    key=q.replace('silog_over_vda_norm_output_conv','silog_over_vda_norm_output_conv')
    out.append((m,s,'train_batches',{'vda_l2_output_conv':'vda_l2_output_conv'}.get(q,q),np.mean(v),len(v)))
# ratio alias
with open('trajectory.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['model','step','dataset','quantity','value','n']); w.writerows(out)
print(len(out))
