"""Overfitting driver. usage: overfit.py KIND(vkitti|tartanair) NB(1|10) HINGE(0|1) STEPS [tag]
Real train._compute_loss_vda via train.compute_loss, FT n46 effective cfg, train.build_optimizer, train.clip_gradients(per_group, 1.0), bf16 autocast, batch=1 clip x 32 frames (accum 1)."""
import sys, time, csv, json, math
from common import *
import models
kind, NB, HINGE, STEPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tag = f"{kind}_nb{NB}_{'hinge' if HINGE else 'base'}"
cfg = load_cfg()
if HINGE:
    cfg['loss'].update(sky_hinge_weight=0.05, sky_hinge_min_depth=400.0, sky_hinge_metric_only=False)
cfg['loss']['log_terms'] = True
dev = 'cuda'
torch.manual_seed(0); np.random.seed(0)
core = models.build_model(cfg).to(dev)
opt = train.build_optimizer(core, cfg)
mods = models.trainable_modules(core)
dh = {id(p) for p in core.depth_head_parameters()}
groups = {'dpt_head': [p for p in core.core.head.parameters() if p.requires_grad and id(p) in dh],
          'motion_modules': [p for n, p in core.core.head.named_parameters() if 'motion_modules' in n and p.requires_grad]}
print({k: sum(p.numel() for p in v) for k, v in groups.items()}, flush=True)
data = torch.load(SCR / f'batches_{kind}.pt')['items'][:NB]
def mk(it):
    b = collate([it])
    b.pop('degraded_frames', None); b.pop('degrade_params', None)
    if kind == 'vkitti': b['silog_ceil'] = torch.tensor([float('inf')])
    return b
batches = [mk(it) for it in data]
amp = torch.bfloat16
lr = float(cfg['training']['learning_rate']); WARM = 10
def fwd(b, grad):
    rgb = b['frames'].to(dev)
    with torch.set_grad_enabled(grad), torch.autocast('cuda', dtype=amp):
        return core(rgb)
@torch.no_grad()
def evaluate(step, keep=None):
    core.eval()
    cats = {'sky': [], 'far80_655': [], 'near<30': [], 'valid_all': []}
    gts = {k: [] for k in cats}
    preds_keep = {}
    for bi, b in enumerate(batches):
        p = fwd(b, False).float().cpu()[:, 0]            # (1,T,H,W)
        d, m, s = b['depth'][:, 0], b['mask'][:, 0] > .5, (b['sky'][:, 0] > .5) & ~(b['mask'][:, 0] > .5)
        for k, sel in (('sky', s), ('far80_655', m & (d >= 80) & (d < 655)), ('near<30', m & (d < 30)), ('valid_all', m)):
            cats[k].append(p[sel]); gts[k].append(d[sel])
        if keep is not None and bi < 2: preds_keep[bi] = p[0].clone()
    out = {'step': step}
    for k in cats:
        pr, gt = torch.cat(cats[k]), torch.cat(gts[k])
        if pr.numel() == 0: continue
        out[f'{k}_pred_med'] = float(pr.median())
        if k != 'sky':
            out[f'{k}_gt_med'] = float(gt.median()); out[f'{k}_ratio_med'] = float((pr / gt).median()); out[f'{k}_absrel'] = float(((pr - gt).abs() / gt).mean())
            out[f'{k}_logbias'] = float((pr.clamp_min(1e-3).log() - gt.log()).mean())
        else:
            out['sky_frac_ge400'] = float((pr >= 400).float().mean()); out['sky_frac_ge200'] = float((pr >= 200).float().mean()); out['sky_pred_p10'] = float(pr.quantile(0.1)) if pr.numel() < 16_000_000 else float(pr[::7].quantile(0.1))
    core.train()
    return out, preds_keep
core.train()
rows, evals = [], []
keep0 = None
t0 = time.time()
ev, keep0 = evaluate(0, keep=True); evals.append(ev); print(json.dumps(ev), flush=True)
for step in range(1, STEPS + 1):
    b = batches[(step - 1) % NB]
    for g in opt.param_groups: g['lr'] = lr * min(1.0, step / WARM)
    train._TERM_ACC.clear()
    pred = fwd(b, True)
    with torch.autocast('cuda', dtype=amp):
        loss = train.compute_loss(pred, b, cfg, pred_is_disparity=False)
    terms = train.term_flush()
    loss.backward()
    gn = {k: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in v if p.grad is not None))) for k, v in groups.items()}
    gn['total'] = math.sqrt(sum(v * v for v in gn.values()))
    train.clip_gradients(core, 1.0, True)
    opt.step(); opt.zero_grad(set_to_none=True)
    row = {'step': step, 'batch': (step - 1) % NB, 'total': float(loss), **{k: terms.get(k, float('nan')) for k in ('vda_total', 'vda_spatial', 'vda_tgm', 'silog', 'silog_mean_log_err', 'sky_hinge_weighted')},
           'gn_dpt_head': gn['dpt_head'], 'gn_motion': gn['motion_modules'], 'gn_total': gn['total'], 'clipped': int(gn['dpt_head'] > 1 or gn['motion_modules'] > 1)}
    rows.append(row)
    if step % 25 == 0:
        ev, _ = evaluate(step); evals.append(ev)
        print(f"[{tag}] step {step} loss {row['total']:.4f} vda {row['vda_total']:.4f} silog {row['silog']:.4f} gn_h {gn['dpt_head']:.3g} gn_m {gn['motion_modules']:.3g} | sky_med {ev['sky_pred_med']:.0f} far_ratio {ev['far80_655_ratio_med']:.3f} near_ratio {ev['near<30_ratio_med']:.3f} ({time.time()-t0:.0f}s) mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
ev, keepN = evaluate(STEPS, keep=True)
# final eval already at step%25 if STEPS%25==0; ensure keepN
for name, rr in (('curves', rows), ('evals', evals)):
    with open(SMALL / f'{tag}_{name}.csv', 'w', newline='') as f:
        keys = sorted({k for r in rr for k in r}, key=lambda k: list(rr[0]).index(k) if k in rr[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rr)
torch.save({'step0': keep0, 'final': keepN, 'idx': [0, 1][:NB]}, SCR / f'{tag}_preds.pt')
print('DONE', tag, time.time() - t0, flush=True)
