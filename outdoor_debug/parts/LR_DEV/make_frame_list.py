"""Builds the DDAD val frame list from tar_listing.txt (file names only, no payload) and ddad.json split 1.
Rule (fixed before any GT/prediction is read): for each val scene (ddad.json scene_splits['1'], 000150..000199),
sort CAMERA_01 png names ascending (timestamp order) and keep indices 0,10,20,... ; the LIDAR datum is paired per the official scene_*.json sample (camera and LIDAR timestamps differ), resolved at extraction.
Sanity frames = index 0 of the first 5 scenes in the split file order."""
import json, re, hashlib, sys, collections
D='/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/LR_DEV'
split=json.load(open(D+'/ddad.json'))['scene_splits']['1']['filenames']
val=[f.split('/')[0] for f in split]
cam=collections.defaultdict(list); lid=collections.defaultdict(set); size=collections.Counter()
for ln in open(D+'/tar_listing.txt'):
    p=ln.split(None,5); 
    if len(p)<6: continue
    name=p[5].rstrip('\n').split(' -> ')[0]; sz=int(p[2]); m=name.split('/')
    if len(m)<3 or m[1] not in val: continue
    sc=m[1]
    if '/rgb/CAMERA_01/' in name and name.endswith('.png'): cam[sc].append(m[-1]); size[sc]+=sz
    elif '/point_cloud/LIDAR/' in name and not name.endswith('/'): lid[sc].add(m[-1].rsplit('.',1)[0]); size[sc]+=sz
    elif m[-1].endswith('.json') and ('/calibration/' in name or m[2].startswith('scene_')): size[sc]+=sz
rows=[]
for sc in val:
    fr=sorted(cam[sc]); ts=[f[:-4] for f in fr]
    for i in range(0,len(fr),10):
        pass
        rows.append((sc,ts[i]))
out=D+'/ddad_val_frames.tsv'
open(out,'w').write(''.join(f"{a}\t{b}\n" for a,b in rows))
print('scenes',len(val),'cam frames',sum(len(v) for v in cam.values()),'selected',len(rows),'sha256',hashlib.sha256(open(out,'rb').read()).hexdigest())
print('subset bytes (CAMERA_01 png + lidar + scene/calibration json, all frames)',sum(size.values()))
print('sanity', [(sc,sorted(cam[sc])[0][:-4]) for sc in val[:5]])
