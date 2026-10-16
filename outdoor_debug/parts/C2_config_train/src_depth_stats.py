import cv2, numpy as np, random, glob, sys
sys.path.insert(0,'/home/wg19671/.claude/jobs/a0d391cb/tmp/od')
from data.pointodyssey_dataset import _PO_DEPTH_SCALE
random.seed(2)
R='/mnt/DATA/DegradedDepth/data/TRAIN'
def stat(name, files, dec, valid_fn, sky_fn=None, n=150):
    fs=random.sample(files,min(n,len(files))); a=np.zeros(7); 
    for f in fs:
        d=dec(f).astype(np.float64); v=valid_fn(d); s=sky_fn(d) if sky_fn else np.zeros_like(v)
        a+=[d.size,v.sum(),s.sum(),(v&(d>=30)).sum(),(v&(d>=80)).sum(),(v&(d>=200)).sum(),(~v&~s).sum()]
    print(f'{name:22s} files={len(files):7d} sampled={len(fs)} valid%={100*a[1]/a[0]:.2f} sky%={100*a[2]/a[0]:.2f} validfar>=30m%={100*a[3]/a[0]:.2f} >=80m%={100*a[4]/a[0]:.2f} >=200m%={100*a[5]/a[0]:.2f} invalid(not valid,not sky)%={100*a[6]/a[0]:.2f}  | of valid px: >=30m {100*a[3]/max(a[1],1):.2f}% >=80m {100*a[4]/max(a[1],1):.2f}%')
vk=glob.glob(R+'/vkitti/*/*/frames/depth/Camera_[01]/depth_*.png')
stat('vkitti (cap655)',vk,lambda f:cv2.imread(f,-1).astype(np.float32)/100.,lambda d:(d>0)&(d<655),lambda d:d>=655)
stat('vkitti (cap80 old B)',vk,lambda f:cv2.imread(f,-1).astype(np.float32)/100.,lambda d:(d>0)&(d<80),lambda d:d>=655)
for split in ['train','val']:
    po=glob.glob(R+f'/point_odyssey/{split}/*/depths/depth_*.png')
    stat(f'PO {split} (cap999)',po,lambda f:cv2.imread(f,-1).astype(np.float32)*_PO_DEPTH_SCALE,lambda d:np.isfinite(d)&(d>0)&(d<999))
    stat(f'PO {split} (cap80?)',po,lambda f:cv2.imread(f,-1).astype(np.float32)*_PO_DEPTH_SCALE,lambda d:np.isfinite(d)&(d>0)&(d<80))
kg=glob.glob(R+'/kitti_raw_gt/train/*/proj_depth/groundtruth/image_02/*.png')
stat('kitti GT (cap80)',kg,lambda f:cv2.imread(f,-1).astype(np.float32)/256.,lambda d:(d>0)&(d<80))
stat('kitti GT (nocap)',kg,lambda f:cv2.imread(f,-1).astype(np.float32)/256.,lambda d:(d>0))
