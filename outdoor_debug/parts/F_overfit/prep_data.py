"""Pick fixed clips (train data only) and cache them as .pt. usage: prep_data.py vkitti|tartanair"""
import sys, json, random
from common import *
kind = sys.argv[1]
ds = make_source(kind)
n = len(ds); print(kind, 'clips', n, flush=True)
random.seed(7)
cand = random.sample(range(n), min(n, 60 if kind == 'tartanair' else 40))
rows = []
for i in cand:
    it = ds[i]
    m, sk, d = it['mask'] > 0.5, it['sky'] > 0.5, it['depth']
    f = lambda x: float(x.float().mean())
    dm = (d > 0)
    row = dict(i=i, vid=it['video_id'], start=int(it['start_frame']) if not torch.is_tensor(it['start_frame']) else int(it['start_frame']),
               sky=f(sk), valid=f(m), far80_655=f(m & (d >= 80) & (d < 655)), far200=f(m & (d >= 200)), band655_1e4=f(m & (d >= 655)),
               near30=f(m & (d < 30)))
    rows.append(row); print(row, flush=True)
json.dump(rows, open(SCR / f'cand_{kind}.json', 'w'))
