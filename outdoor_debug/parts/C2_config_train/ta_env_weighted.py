import cv2, numpy as np, random, glob, collections
random.seed(3)
root='/mnt/DATA/DegradedDepth/data/TRAIN/tartanair'
OUT={'AmusementPark','BrushifyMoon','DesertGasStation','Downtown','ForestEnv','GreatMarsh','HongKong','JapaneseCity','ModernCityDowntown','NordicHarbor','Ocean','PolarSciFi','Rome','SeasonalForestWinter','WesternDesertTown','ConstructionSite','AbandonedFactory'}
envs=sorted(p.split('/')[-1] for p in glob.glob(root+'/*') if not p.endswith('.zip'))
cnt={};st={}
for e in envs:
    fs=glob.glob(f'{root}/{e}/Data_*/P*/depth_lcam_front/*_depth.png')
    if not fs: continue
    cnt[e]=len(fs); a=np.zeros(7)
    for f in random.sample(fs,min(40,len(fs))):
        d=np.squeeze(np.ascontiguousarray(cv2.imread(f,-1)).view('<f4'),-1).astype(np.float64)
        fin=np.isfinite(d); v=fin&(d>0)&(d<1e4); s=(~fin)|(d>1e4)
        a+=[d.size,v.sum(),s.sum(),(v&(d>=30)).sum(),(v&(d>=80)).sum(),(v&(d>=200)).sum(),(v&(d>=655)).sum()]
    st[e]=a/a[0]
T=sum(cnt.values()); 
def agg(sel):
    w=np.array([cnt[e] for e in sel],float); w/=w.sum()
    return sum(wi*st[e] for wi,e in zip(w,sel))
for lab,sel in [('ALL',list(cnt)),('outdoor-labelled',[e for e in cnt if e in OUT]),('indoor-labelled',[e for e in cnt if e not in OUT])]:
    r=agg(sel); share=sum(cnt[e] for e in sel)/T
    print(f'TA {lab:17s} frame-share={share:.3f} valid={r[1]:.4f} sky={r[2]:.4f} validfar>=30={r[3]:.4f} >=80={r[4]:.4f} >=200={r[5]:.4f} >=655={r[6]:.4f}')
