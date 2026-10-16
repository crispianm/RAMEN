"""Instantiate the REAL training dataset classes (train._make_source, C3 spec, local roots) on local train data (CPU),
sample clips, and log per-source pixel fractions after preprocessing + sky contribution per loss term.
usage: stats.py NCLIPS NFRAMES"""
import sys, json, random, pathlib, collections
import numpy as np, torch
sys.path.insert(0, '.')
import train
T = pathlib.Path('/mnt/DATA/DegradedDepth/data/TRAIN')
OUT = pathlib.Path('/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/B2_transforms_masks')
cfg = json.load(open('outdoor_debug/parts/B2_transforms_masks/effective_C3.json'))
d = cfg['data']
NCL, NF = int(sys.argv[1]), int(sys.argv[2])
roots = {'tartanair': T/'tartanair', 'vkitti': T/'vkitti', 'pointodyssey_train': T/'point_odyssey/train', 'pointodyssey_val': T/'point_odyssey/val',
         'irs': T/'irs', 'spring': T/'spring', 'kitti_raw': T/'kitti_raw', 'scannetpp_train': T/'scannetpp_train_full/data'}
common = dict(num_frames=NF, image_size=(518, 518), clip_stride=int(d['clip_stride']), frame_stride=1, max_depth=float(d['max_depth']), seed=45,
              degrade=True, degrade_mode=['awgn_blur'], degrade_severity=0.0, max_clips_per_video=None, resize_mode='vda')
rows = {}
def run(name, spec):
    spec = dict(spec)
    if 'root' in spec: spec['root'] = str(roots[name])
    if 'gt_root' in spec: spec['gt_root'] = str(T/'kitti_raw_gt')
    if 'po_keep_list' in spec:   # local copy of the keep list restricted to scenes present on this host (the sha pin cannot hold for the filtered copy)
        kl = json.load(open(spec['po_keep_list'])); root = pathlib.Path(spec['root'])
        kl['kept'] = [k for k in kl['kept'] if (root/k).is_dir()]
        if not kl['kept']: rows[name] = {'error': 'no keep-list scene present locally'}; print(name, 'SKIP no local scenes'); return
        out = OUT/('po_keep_local_%s.json' % name); json.dump(kl, open(out, 'w')); spec['po_keep_list'] = str(out); spec.pop('po_keep_sha256', None)
        print(name, 'local keep scenes', len(kl['kept']))
    try:
        ds = train._make_source(spec, 'train', common)
    except Exception as e:
        print(name, 'INSTANTIATE FAIL', repr(e)[:300]); rows[name] = {'error': repr(e)[:300]}; return
    n = len(ds); print(name, 'clips', n, 'class', type(ds).__name__, flush=True)
    if n == 0: rows[name] = {'error': 'no clips'}; return
    random.seed(1); torch.manual_seed(1)
    idxs = random.sample(range(n), min(NCL, n))
    acc = collections.Counter(); samples = []
    ceil = spec.get('silog_max_depth'); msup = spec.get('metric_supervision', False)
    for i in idxs:
        try: it = ds[i]
        except Exception as e:
            print(' item fail', i, repr(e)[:200]); acc['fail'] += 1; continue
        dep, m, sk = it['depth'].float(), it['mask'].float(), it['sky'].float()
        tot = m.numel()
        valid, sky = m > 0.5, sk > 0.5
        acc['n_clip'] += 1; acc['px'] += tot
        acc['valid'] += int(valid.sum()); acc['sky'] += int(sky.sum())
        acc['invalid_nonsky'] += int((~valid & ~sky).sum())
        acc['valid_and_sky'] += int((valid & sky).sum())
        acc['far80'] += int((valid & (dep > 80)).sum()); acc['far200'] += int((valid & (dep > 200)).sum())
        acc['mask_nonbinary'] += int(((m != 0) & (m != 1)).sum()); acc['sky_nonbinary'] += int(((sk != 0) & (sk != 1)).sum())
        acc['depth_nonfinite'] += int((~torch.isfinite(dep)).sum()); acc['depth_neg'] += int((dep < 0).sum())
        acc['valid_depth_le0'] += int((valid & (dep <= 0)).sum())
        acc['valid_depth_ge_bound'] += int((valid & (dep >= float(spec.get('max_depth', d['max_depth'])))).sum())
        # depth values stored under sky / invalid
        if sky.any(): acc['sky_depth_sum'] += float(dep[sky].sum()); acc['sky_depth_cnt'] += int(sky.sum()); acc['sky_depth_max'] = max(acc['sky_depth_max'], float(dep[sky].max()))
        inv = ~valid & ~sky
        if inv.any(): acc['inv_depth_nonzero'] += int((dep[inv] != 0).sum())
        # loss contributions (identical to train.py): VDA SSI/grad/TGM mask m_loss = m | sky (sky_vda_zero_disparity)
        m_loss = valid | sky
        acc['vda_terms_px_valid'] += int(valid.sum()); acc['vda_terms_px_sky_at_disp0'] += int((sky & ~valid).sum())
        sup = bool(msup)
        silog_px = valid & (dep < (ceil if ceil else float('inf'))) if sup else torch.zeros_like(valid)
        acc['silog_px'] += int(silog_px.sum()); acc['silog_px_sky'] += int((silog_px & sky).sum())
        acc['hinge_px'] += 0  # sky_hinge_weight = 0 in C3
        if len(samples) < 6: samples.append(i)
    P = acc['px'] or 1
    r = {k: acc[k] for k in acc}
    r.update(dict(pct_valid=100*acc['valid']/P, pct_sky=100*acc['sky']/P, pct_invalid_nonsky=100*acc['invalid_nonsky']/P,
                  pct_far_gt80_of_all=100*acc['far80']/P, pct_far_gt200_of_all=100*acc['far200']/P,
                  pct_far_gt80_of_valid=100*acc['far80']/max(acc['valid'], 1),
                  sky_stored_depth_mean=acc['sky_depth_sum']/max(acc['sky_depth_cnt'], 1),
                  pct_sky_in_vda_loss_mask=100*acc['vda_terms_px_sky_at_disp0']/max(acc['vda_terms_px_sky_at_disp0']+acc['vda_terms_px_valid'], 1),
                  sparse_gt=spec.get('sparse_gt'), metric_sup=spec.get('metric_supervision'), silog_ceil=ceil, max_depth=spec.get('max_depth', d['max_depth']),
                  vkitti_max_depth=spec.get('vkitti_max_depth'), n_source_clips=n, clip_class=type(ds).__name__))
    rows[name] = r
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k.startswith('pct') or k in ('mask_nonbinary', 'sky_nonbinary', 'valid_and_sky', 'depth_nonfinite', 'inv_depth_nonzero', 'valid_depth_le0', 'sky_depth_max', 'sky_stored_depth_mean', 'n_clip')}), flush=True)

names = []
for s in d['sources']:
    t = s['type']
    nm = {'pointodyssey': 'pointodyssey_train' if s['root'].endswith('/train') else 'pointodyssey_val'}.get(t, t)
    if nm in ('nyu_train', 'scannetpp_train'): continue
    names.append((nm, s))
for nm, s in names: run(nm, s)
json.dump(rows, open('outdoor_debug/parts/B2_transforms_masks/stats_%d_%d.json' % (NCL, NF), 'w'), indent=1)
