import cv2, numpy as np, random, glob, collections, json
random.seed(0)
root='/mnt/DATA/DegradedDepth/data/TRAIN/tartanair'
envs=sorted(p.split('/')[-1] for p in glob.glob(root+'/*') if not p.endswith('.zip'))
agg=collections.defaultdict(lambda: np.zeros(8)); res={}
for e in envs:
    fs=glob.glob(f'{root}/{e}/Data_*/P*/depth_lcam_front/*_depth.png')
    if not fs: continue
    for f in random.sample(fs,min(12,len(fs))):
        b=cv2.imread(f,cv2.IMREAD_UNCHANGED)
        d=np.squeeze(np.ascontiguousarray(b).view('<f4'),-1).astype(np.float64)
        n=d.size
        fin=np.isfinite(d)
        valid=fin&(d>0)&(d<10000)
        sky=(~fin)|(d>1e4)
        neither=~valid&~sky
        agg[e]+=np.array([n,valid.sum(),sky.sum(),neither.sum(),(valid&(d>=80)).sum(),(valid&(d>=655)).sum(),(d==10000).sum(), (fin&(d>1e4)).sum()])
        if e=='x': pass
mx=0
tot=np.zeros(8)
print('env,valid%,sky%,neither%,valid>=80m%,valid>=655m%,==10000%,finite>1e4%')
for e,a in agg.items():
    tot+=a
    print(e,*[f'{100*x/a[0]:.2f}' for x in a[1:]],sep=',')
print('ALL(equal-weight-per-sample)',*[f'{100*x/tot[0]:.2f}' for x in tot[1:]],sep=',')
