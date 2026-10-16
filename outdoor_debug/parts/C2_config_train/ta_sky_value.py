import cv2, numpy as np, random, glob
random.seed(1)
root='/mnt/DATA/DegradedDepth/data/TRAIN/tartanair'
fs=glob.glob(f'{root}/*/Data_*/P*/depth_lcam_front/*_depth.png'); fs=random.sample(fs,400)
sky=[];edges=np.array([0,10,30,80,200,655,2000,9999.99])
h=np.zeros(len(edges)-1)
for f in fs:
    d=np.squeeze(np.ascontiguousarray(cv2.imread(f,-1)).view('<f4'),-1)
    s=d[d>1e4]; 
    if s.size: sky.append(np.unique(np.round(s,1))[:5])
    v=d[(d>0)&(d<1e4)]; h+=np.histogram(v,edges)[0]
u=np.unique(np.concatenate(sky)); print('distinct sky values (first):',u[:10], 'n distinct',len(u))
print('valid-pixel depth histogram edges',edges.tolist()); print('fractions',(h/h.sum()).round(4).tolist())
