"""Phase 9: effective source probabilities, sampler simulation, expected pixel statistics of the r50 / C3 mix."""
import sys, glob, random, collections, json
import numpy as np, torch, cv2, yaml
sys.path.insert(0,'/home/wg19671/.claude/jobs/a0d391cb/tmp/od')
from pathlib import Path
import train
from torch.utils.data import WeightedRandomSampler
cfg=train.load_config(Path('/home/wg19671/.claude/jobs/a0d391cb/tmp/od/configs/experiments/dn01_c3rep_20261009_c3_n46.yaml'))
S=cfg['data']['sources']
names=['tartanair','vkitti','pointodyssey:train','pointodyssey:val','irs','spring','kitti_raw','nyu_train','scannetpp_train']
clips=dict(zip(names,[9781,1280,7204,1059,3191,134,1277,1307,54585]))   # provenance/dn01_v3/synreal_counts.json (clips at clip_stride 32)
mass=dict(zip(names,[s['sampling_mass'] for s in S])); tot=sum(mass.values())
p={k:mass[k]/tot for k in names}
print('sampling:',cfg['data']['sampling'],'| total mass',tot)
# 1) simulation exactly as _build_multisource: weights = mass/len per clip, WeightedRandomSampler(replacement=True), world_size 4
weights=[]; src_of=[]
for k in names:
    weights += [mass[k]/clips[k]]*clips[k]; src_of += [k]*clips[k]
src_of=np.array(src_of)
W=4; counts=collections.Counter(); clipcount=collections.Counter()
N=10000
for rank in range(W):
    g=torch.Generator().manual_seed(46*1000+rank+250)
    smp=WeightedRandomSampler(weights,num_samples=len(weights)//W,replacement=True,generator=g)
    idx=np.array(list(iter(smp)))
    idx10=idx[:N//W*1]  # first N/4 per rank -> 10k draws over 4 ranks
    for k in src_of[idx10]: counts[k]+=1
    clipcount.update(idx10.tolist())
tot_d=sum(counts.values())
rows=[]
print(f'{"source":22s} mass_share  sim_share(10k draws, 4 ranks)  clips  expected_repeats_over_12k_steps(192k clips)')
for k in names:
    rep=192000*p[k]/clips[k]
    print(f'{k:22s} {p[k]:.4f}      {counts[k]/tot_d:.4f}                      {clips[k]:6d}  {rep:7.2f}')
    rows.append((k,mass[k],p[k],counts[k]/tot_d,clips[k],rep))
print('epoch length per rank (num_samples) =',len(weights)//W,'-> optimizer steps/epoch (bs1,accum4) =',len(weights)//W//4,'; 12000 steps = %.2f epochs'%(12000/(len(weights)//W//4)))
print('seed streams per rank:',[46*1000+r+250 for r in range(W)],'(x1000 scheme: global seeds',[46*1000+r for r in range(W)],')')
# 2) per-source pixel fractions (measured on local DEV/train data, sample of frames; see *_stats.txt) 
# values: (valid, sky, validfar>=30m, >=80m, >=200m) as fraction of ALL pixels
meas={
 'vkitti':(0.8383,0.1617,0.2459,0.0733,0.0102),
 'pointodyssey:train':(0.8590,0.0,0.0091,0.0001,0.0),
 'pointodyssey:val':(0.9064,0.0,0.0027,0.0,0.0),
 'kitti_raw':(0.1537,0.0,0.0134,0.0,0.0),
}
ta_env=None
json.dump({'rows':rows},open('mix_stats.json','w'))

# ---- combined expectations (TA from ta_env_weighted.txt ALL row; others from src_depth_stats.txt) ----
TA=(0.9151,0.0848,0.1567,0.0927,0.0689)           # valid, sky, >=30, >=80, >=200 (fractions of all pixels), env-frame-weighted
meas['tartanair']=TA
tot_px=lambda i: sum(p[k]*meas[k][i] for k in meas)
print('\nExpected fractions of ALL pixels of a sampled batch (only sources with measured values; IRS/spring/NYU/ScanNet++ treated as 0 far, 0 sky):')
for i,lab in enumerate(['valid','sky-labelled','valid >=30 m','valid >=80 m','valid >=200 m']):
    print(f'  {lab:14s} {100*tot_px(i):.2f}%')
sky_src=['tartanair','vkitti']
print('clip share of sources that carry a sky label (TA+VKITTI): %.1f%%'%(100*sum(p[k] for k in sky_src)))
print('clip share of sources with NO sky label (PO, IRS, spring, KITTI, NYU, ScanNet++): %.1f%%'%(100*(1-sum(p[k] for k in sky_src))))
ind=['irs','nyu_train','scannetpp_train']; print('indoor-only sources (IRS+NYU+ScanNet++): %.1f%%'%(100*sum(p[k] for k in ind)))
out_def=p['kitti_raw']+p['vkitti']; print('definitely-outdoor sources (KITTI+VKITTI): %.1f%%'%(100*out_def))
ta_out=0.755*p['tartanair']; print('TartanAir outdoor-labelled envs (75.5%% of frames): %.1f%% of clips; TA indoor-labelled %.1f%%'%(100*ta_out,100*(p['tartanair']-ta_out)))
print('metric-supervised (SiLog) clip share: %.1f%% (everything but Spring)'%(100*(1-p['spring'])))
print('metric-supervised OUTDOOR clip share (KITTI+VKITTI+TA outdoor envs; PO unknown): %.1f%%'%(100*(out_def+ta_out)))
# SiLog-eligible far pixels: TA/PO <655 m; VKITTI <655; KITTI<=80 sparse
print('SiLog-eligible px >=80 m: TA (valid>=80 minus >=655) %.2f%% of TA px; VKITTI %.2f%% of VKITTI px'%(100*(0.0927-0.0414),100*0.0733))
print('mix share of valid >=80 m px by source:',{k:round(100*p[k]*meas[k][3],3) for k in meas})
print('mix share of sky px by source:',{k:round(100*p[k]*meas[k][1],3) for k in meas})
