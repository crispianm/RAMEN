import sys, csv, json
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common import SMALL, SCR
def rd(p): return list(csv.DictReader(open(p)))
def col(rows, k): return np.array([float(r[k]) if r.get(k, '') not in ('', 'nan') else np.nan for r in rows])
def sm(x, w=9): 
    return np.convolve(np.pad(x, (w//2, w//2), mode='edge'), np.ones(w)/w, mode='valid')
for kind in ('vkitti', 'tartanair'):
    for nb in (1, 10):
        fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
        for hinge, ls in ((0, '-'), (1, '--')):
            tag = f"{kind}_nb{nb}_{'hinge' if hinge else 'base'}"
            try: c, e = rd(SMALL / f'{tag}_curves.csv'), rd(SMALL / f'{tag}_evals.csv')
            except FileNotFoundError: continue
            lab = 'hinge 0.05@400m' if hinge else 'baseline (hinge 0)'
            s = col(c, 'step')
            ax[0,0].plot(s, sm(col(c, 'total')), ls, label=lab+' total')
            ax[0,0].plot(s, sm(col(c, 'silog')), ls, alpha=.6, label=lab+' silog')
            ax[0,0].set_yscale('log'); ax[0,0].set_title('loss (9-step mean)')
            ax[0,1].plot(s, sm(col(c, 'vda_spatial')), ls, label=lab+' SSI+grad'); ax[0,1].plot(s, sm(col(c, 'vda_tgm')), ls, alpha=.6, label=lab+' TGM*'); ax[0,1].set_yscale('log'); ax[0,1].set_title('VDA terms (spatial, tgm)')
            ax[0,2].plot(s, col(c, 'gn_dpt_head'), ls, label=lab+' dpt_head'); ax[0,2].plot(s, col(c, 'gn_motion'), ls, alpha=.6, label=lab+' motion'); ax[0,2].set_yscale('log'); ax[0,2].axhline(1, color='k', lw=.5); ax[0,2].set_title('pre-clip grad norm (clip=1.0)')
            es = col(e, 'step')
            ax[1,0].plot(es, col(e, 'sky_pred_med'), ls, label=lab+' sky median'); ax[1,0].set_title('predicted depth over SKY pixels (m)'); ax[1,0].set_yscale('log')
            ax[1,1].plot(es, col(e, 'far80_655_ratio_med'), ls, label=lab); ax[1,1].axhline(1, color='k', lw=.5); ax[1,1].set_title('far valid 80-655 m: median pred/gt')
            ax[1,2].plot(es, col(e, 'near<30_ratio_med'), ls, label=lab); ax[1,2].axhline(1, color='k', lw=.5); ax[1,2].set_title('near valid <30 m: median pred/gt')
        for a in ax.ravel(): a.set_xlabel('step'); a.legend(fontsize=7); a.grid(alpha=.3)
        fig.suptitle(f'{kind}, {nb} fixed clip(s) x 32 frames, FT n46 recipe'); fig.tight_layout()
        fig.savefig(SMALL / f'curves_{kind}_nb{nb}.png', dpi=60); plt.close(fig)
    # panels
    B = torch.load(SCR / f'batches_{kind}.pt')['items']
    for nb in (1, 10):
        for hinge in (0, 1):
            tag = f"{kind}_nb{nb}_{'hinge' if hinge else 'base'}"
            try: P = torch.load(SCR / f'{tag}_preds.pt')
            except FileNotFoundError: continue
            keys = list(P['step0'].keys())
            fig, ax = plt.subplots(len(keys), 5, figsize=(20, 3.6 * len(keys)))
            ax = np.atleast_2d(ax)
            for r, bi in enumerate(keys):
                it = B[bi]; f = 16
                rgb = it['frames'][:, f].permute(1, 2, 0).numpy().clip(0, 1)
                gt = it['depth'][0, f].numpy().copy(); sk = (it['sky'][0, f] > .5) & ~(it['mask'][0, f] > .5)
                gt[sk.numpy()] = 700.0   # sky target shown at the cap value
                gtm = np.where(it['mask'][0, f].numpy() > .5, gt, np.where(sk.numpy(), 700.0, np.nan))
                p0, pN = P['step0'][bi][f].numpy(), P['final'][bi][f].numpy()
                ims = [rgb, gtm, p0, pN]
                ax[r,0].imshow(rgb); ax[r,0].set_title(f'RGB clip{bi} f{f}')
                for j, (im, t) in enumerate(((gtm, 'GT (sky shown 700, nan=invalid)'), (p0, 'pred step 0 (released)'), (pN, 'pred end'))):
                    h = ax[r,j+1].imshow(np.clip(im, 1, 700), cmap='turbo_r', norm=LogNorm(1, 700)); ax[r,j+1].set_title(t)
                    ax[r,j+1].contour(sk.numpy().astype(float), levels=[.5], colors='w', linewidths=.8)
                # sky median text
                skn = sk.numpy()
                if skn.any(): ax[r,2].text(5, 20, f'sky med {np.median(p0[skn]):.0f} m', color='w', fontsize=9, bbox=dict(fc='k', alpha=.5)); ax[r,3].text(5, 20, f'sky med {np.median(pN[skn]):.0f} m', color='w', fontsize=9, bbox=dict(fc='k', alpha=.5))
                err = np.abs(pN - np.nan_to_num(gtm, nan=1)) / np.nan_to_num(gtm, nan=1)
                ax[r,4].imshow(np.where(np.isnan(gtm), np.nan, np.log(np.clip(pN,1e-3,None) / np.nan_to_num(gtm, nan=1))), cmap='RdBu_r', vmin=-1, vmax=1); ax[r,4].set_title('log(pred_end/gt)  red=too far')
                for a in ax[r]: a.axis('off')
            fig.suptitle(tag + ' (white outline = sky label; log colour scale 1-700 m)'); fig.tight_layout()
            fig.savefig(SCR / f'panel_{tag}_full.png', dpi=80); fig.savefig(SMALL / f'panel_{tag}.jpg', dpi=45, pil_kwargs=dict(quality=70)); plt.close(fig)
