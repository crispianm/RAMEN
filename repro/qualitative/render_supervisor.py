"""C-010: DEV-only inference; all writes contained in this task folder.
Adapted from copied render_c3_figures.py and perframe_core.py. No table scoring.
"""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('RAMEN_QUAL_OUT', HERE))   # VENDORED: output root (scratch/, media/) is now an env var; was the script dir
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MPLCONFIGDIR'] = str(ROOT/'scratch/mpl')
os.environ['XDG_CACHE_HOME'] = str(ROOT/'scratch/cache')
os.environ['TMPDIR'] = str(ROOT/'scratch/tmp')
import sys, json, time, hashlib, subprocess, gc
sys.dont_write_bytecode = True
import numpy as np
import cv2
cv2.setNumThreads(2)
CFG = json.loads((HERE/'figures_config.json').read_text())
sys.path.insert(0, CFG['eval_clone'])
sys.path.insert(0, str(HERE))
import torch
import bench_joint as bj
import perframe_core as core
import render_c3_figures as rf
torch.set_num_threads(4)
SELFILE = HERE/'CLIP_SELECTION.json'
SELECTED = {s['dataset']:s['clip'] for s in json.loads(SELFILE.read_text())['selected'] if s['kind']=='median'}
SETS = ['kitti','bonn','scannetpp']
SPECS = {'ours': 'ours:'+CFG['c3']['ckpt'], 'ft': bj.FT_VDA+CFG['ft']['B']['ckpt'], 'released':'metric_vda_s'}
NAMES = {'ours':'Ours + TAG (seed 45)', 'ft':'Fine-tuned Metric-VDA-S (seed 46)', 'released':'Released Metric-VDA-S'}
INPUTS = set([str(SELFILE), __file__, CFG['eval_clone']+'/bench_joint.py'])

def savejson(path, data):
    path.write_text(json.dumps(data, indent=2))

def small(a, w=384, nearest=False):
    h = max(2, round(a.shape[0]*w/a.shape[1]/2)*2)
    return cv2.resize(a,(w,h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

def prepare(ds, key, cell):
    if ds == 'horse':
        paths = sorted(Path('/mnt/DATA/Horse/test_resized').glob('*'))
        paths = [p for p in paths if p.suffix.lower() in ('.png','.jpg','.jpeg')]
        assert len(paths)==51, len(paths)
        INPUTS.update(map(str, paths))
        bj._vda_path()
        from video_depth_anything.util.transform import Resize
        resize = Resize(width=518,height=518,resize_target=False,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC)
        rgb = [cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in paths]
        x = torch.from_numpy(np.stack([np.clip(resize({'image':r.astype(np.float32)/255})['image'],0,1) for r in rgb])).permute(0,3,1,2).contiguous()
        return x,None,rgb[0].shape[:2],{'frames':len(paths),'noise_seed':None,'cell':'real','paths':list(map(str,paths))}
    frames = dict(bj.load_clips(ds,'dev'))[key]
    INPUTS.add(str(bj.PATHS[ds+'_manifest']))
    used = frames[:bj.DATASETS[ds]['max_len']]
    if cell=='st_llvar':
        used=[used[len(used)//2]]
    for f in used:
        INPUTS.add(str(Path(bj.PATHS[ds+'_root'])/f['image']))
        INPUTS.add(str(Path(bj.PATHS[ds+'_root'])/f['gt_depth']))
    clean, hw = bj.load_frames(ds,used)
    gt = bj.load_gt(ds,used)
    if cell=='st_llvar':
        clean,gt=bj.static_clip(clean,gt)
        seed=bj.noise_seed(ds,key,bj.seed_cell(cell))
        x=bj.degrade_static(clean,cell,seed)
        drawn=None
    else:
        seed=bj.llvar_noise_seed(ds,key,cell)
        drawn=bj.llvar_draw(ds,key)
        x=bj.degrade_llvar(clean,cell,drawn,seed)
    return x,gt,hw,{'frames':len(x),'noise_seed':seed,'llvar_draw':drawn,'cell':cell,'paths':[f['image'] for f in used]}

def inference():
    assert subprocess.check_output(['git','-C',CFG['eval_clone'],'rev-parse','HEAD'],text=True).strip()==CFG['eval_commit']
    for c in (CFG['c3'],CFG['ft']['B']):
        assert rf.sha(c['ckpt'])==c['sha256']
    begin=time.time()
    tasks=[(ds,SELECTED[ds],'llvar') for ds in SETS]+[('horse','horse','real')]
    tasks += [(ds,k,'st_llvar') for ds in SETS for k,_ in bj.load_clips(ds,'dev')]
    for model,spec in SPECS.items():
        print('BUILD',model,flush=True)
        p=bj.build_pipeline(spec,'cuda')
        INPUTS.update(p['weights'].values())
        if model=='released':
            assert any(rf.sha(v).startswith('3c28432b') for v in p['weights'].values())
        for ds,key,cell in tasks:
            stem=f'{ds}__{key}__{cell}'
            dest=ROOT/'scratch'/f'{stem}__{model}.json'
            selected = ds=='horse' or key==SELECTED.get(ds)
            if dest.exists() and (not selected or (ROOT/'scratch'/f'{stem}__{model}.npy').exists()):
                continue
            x,gt,hw,meta=prepare(ds,key,cell)
            blur,sigma = bj.parse_cell(cell) if cell!='real' else (0.,0.)
            torch.cuda.synchronize(); t=time.time()
            depth,restored=bj.run_pipeline(p,x.to('cuda'),sigma,blur,hw)
            if model=='ours':
                depth=bj.apply_variant(CFG['tag'],depth,p['last_guides'])
            torch.cuda.synchronize(); secs=time.time()-t
            result={'dataset':ds,'clip':key,'model':model,'split':'real footage; no GT' if ds=='horse' else 'DEV','gpu':'RTX 4090','inference_seconds':secs,**meta}
            if cell=='st_llvar':
                proto=bj.DATASETS[ds]
                # The fixed user definition only; no benchmark table scores are computed.
                d,_,_,_,_=core.prep_depth(depth,gt,proto['max_depth'],proto['min_depth'])
                v=(gt[0]>proto['min_depth']) & (gt[0]<proto['max_depth'])
                result['relcurpct_un_pv']=[float((100*np.abs(d[t]-d[t-1])/d[t])[v].mean()) for t in range(1,len(d))]
                del d
            if selected:
                np.save(ROOT/'scratch'/f'{stem}__{model}.npy',depth)
                if model=='ours':
                    inp=np.stack([small(rf.u8(z)) for z in x])
                    np.save(ROOT/'scratch'/f'{stem}__input.npy',inp)
                    if gt is not None:
                        np.save(ROOT/'scratch'/f'{stem}__gt.npy',gt)
            savejson(dest,result)
            print('DONE',model,ds,key,cell,round(secs,1),'s',flush=True)
            del depth,restored,x,gt
            p.pop('last_guides',None)
            gc.collect();torch.cuda.empty_cache()
            savejson(ROOT/'scratch/inference_inputs.json',sorted(INPUTS))
        del p;gc.collect();torch.cuda.empty_cache()
    old=json.loads((ROOT/'scratch/inference_runs.json').read_text()) if (ROOT/'scratch/inference_runs.json').exists() else []
    old.append({'start_unix':begin,'end_unix':time.time(),'job_wall_seconds':time.time()-begin})
    savejson(ROOT/'scratch/inference_runs.json',old)
    print('GPU FINISHED',flush=True)

def encode(path, frames, w, h):
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','10','-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for f in frames: proc.stdin.write(f.tobytes())
    proc.stdin.close()
    assert proc.wait()==0

def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries=[]
    for ds,key,cell in [(ds,SELECTED[ds],cell) for ds in SETS for cell in ['llvar','st_llvar']]+[('horse','horse','real')]:
        stem=f'{ds}__{key}__{cell}'
        if not all((ROOT/'scratch'/f'{stem}__{m}.json').exists() for m in SPECS):
            print('NOT YET READY',stem,flush=True)
            continue
        inp=np.load(ROOT/'scratch'/f'{stem}__input.npy')
        gp=ROOT/'scratch'/f'{stem}__gt.npy'
        gt=np.load(gp) if gp.exists() else None
        pred={m:np.load(ROOT/'scratch'/f'{stem}__{m}.npy',mmap_mode='r') for m in SPECS}
        maxd=bj.DATASETS[ds]['max_depth'] if ds!='horse' else None
        if gt is not None:
            valid=(gt>1e-3)&(gt<maxd)
            lo,hi=map(float,np.percentile(gt[valid],[2,98]))
        else:
            # Shared range, all models included, never per-model normalization.
            vals=np.concatenate([np.asarray(d)[:,::8,::8].reshape(-1) for d in pred.values()])
            lo,hi=map(float,np.percentile(vals[np.isfinite(vals)&(vals>0)],[2,98]))
        w,h=384,inp.shape[1]
        head=30; left=284; totalw=left+3*w; totalh=4*(h+head)
        def panel_text(canvas,text,pos,size=.58):
            cv2.putText(canvas,text,pos,cv2.FONT_HERSHEY_SIMPLEX,size,(33,43,55),1,cv2.LINE_AA)
        def frame(t):
            out=np.full((totalh,totalw,3),247,np.uint8)
            panel_text(out,f'{ds.upper()} | {cell}',(8,23))
            panel_text(out,f'Frame {t} / {len(inp)-1}',(8,50))
            for j,label in enumerate(['Degraded input' if gt is not None else 'Real low-light input','Ground-truth depth' if gt is not None else 'No ground truth','Shared depth scale']):
                panel_text(out,label,(left+j*w+5,23))
            out[head:head+h,left:left+w]=inp[t]
            if gt is not None:
                gg=rf.depth_rgb(gt[t],lo,hi);gg[~valid[t]]=(70,70,70)
                out[head:head+h,left+w:left+2*w]=cv2.resize(gg,(w,h),interpolation=cv2.INTER_NEAREST)
            else:
                panel_text(out,'Real low-light footage',(left+w+15,head+50))
            panel_text(out,f'{lo:.2f} to {hi:.2f} metres',(left+2*w+12,head+30))
            strip=rf.depth_rgb(np.tile(np.geomspace(max(lo,1e-6),hi,w-24),(22,1)),lo,hi)
            out[head+44:head+66,left+2*w+12:left+3*w-12]=strip
            panel_text(out,'Log colour scale; all rows',(left+2*w+12,head+95),.5)
            for i,(m,d) in enumerate(pred.items(),1):
                y=i*(h+head)
                labels={'ours':['Ours + TAG','C3 seed 45, continuation'], 'ft':['Fine-tuned Metric-VDA-S','Seed 46, from scratch'], 'released':['Released Metric-VDA-S','Released checkpoint']}
                for li,line in enumerate(labels[m]):
                    panel_text(out,line,(8,y+28+li*25),.5)
                for j,label in enumerate(['Metric depth','AbsRel error (0-0.3)','Frame change (0-3%)']):
                    panel_text(out,label,(left+j*w+5,y+23))
                raw=np.asarray(d[t],dtype=np.float64)
                dep=rf.depth_rgb(raw,lo,hi)
                if gt is not None:
                    dd=cv2.resize(raw,(gt.shape[2],gt.shape[1]))
                    er=rf.abs_rel_map(dd,gt[t],valid[t])
                    err=rf.err_rgb(er,.3)
                else:
                    err=np.full_like(dep,247)
                if t:
                    change=np.abs(raw-np.asarray(d[t-1],dtype=np.float64))/np.maximum(raw,1e-3)
                    ch=rf.colour(change/.03,'inferno')
                else: ch=np.full_like(dep,70)
                for j,img in enumerate([dep,err,ch]):
                    out[y+head:y+head+h,left+j*w:left+(j+1)*w]=cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
                if gt is None: panel_text(out,'Not available',(left+w+20,y+head+45))
            return out
        name=f'{ds}_{cell}'
        poster=frame(len(inp)//2)
        cv2.imwrite(str(ROOT/'media'/f'{name}.jpg'),poster[:,:,::-1])
        if '--force' in sys.argv or not (ROOT/'media'/f'{name}.mp4').exists():
            encode(ROOT/'media'/f'{name}.mp4',(frame(t) for t in range(len(inp))),totalw,totalh)
        entry={'dataset':ds,'clip':key,'cell':cell,'frames':len(inp),'fps':10,'depth_range_m':[lo,hi],'error':'unaligned metric |d-g|/g; GT valid; 0–0.3','change':'unaligned |d_t-d_(t-1)|/max(d_t,0.001); 0–3%; no motion compensation','file':f'media/{name}.mp4','poster':f'media/{name}.jpg','split':'DEV' if ds!='horse' else 'real footage, no GT'}
        entries.append(entry)
        print('ENCODED',name,flush=True)
    savejson(ROOT/'media/VIDEOS.json',entries)
    fig,ax=plt.subplots(figsize=(10.5,4.3),layout='constrained')
    colors={'kitti':'#167c80','bonn':'#b25529','scannetpp':'#6550a5'}
    curves=[]
    for ds in SETS:
        for m,ls in [('ours','-'),('ft','--'),('released',':')]:
            files=sorted((ROOT/'scratch').glob(f'{ds}__*__st_llvar__{m}.json'))
            records=[json.loads(f.read_text()) for f in files]
            if len(records)!=len(bj.load_clips(ds,'dev')):
                print('STATIC PLOT awaiting full DEV coverage',ds,m,flush=True)
                plt.close(fig)
                return
            series=np.asarray([r['relcurpct_un_pv'] for r in records])
            mean=series.mean(0);t=np.arange(1,len(mean)+1)
            ax.plot(t,mean,color=colors[ds],linestyle=ls,linewidth=1.8,label=f'{ds} / {m}')
            curves.append({'dataset':ds,'model':m,'num_clips':len(records),'frame_index':t.tolist(),'mean_depth_change_pct':mean.tolist(),'clip_ids':[r['clip'] for r in records]})
    ax.axvspan(1,7,color='#e8e8e8',label='Warm-up (1–7)')
    ax.set(xlabel='Frame index (0-based)',ylabel='Mean relative depth change (%)',xlim=(1,63),ylim=(0,None))
    from matplotlib.lines import Line2D
    legend=[Line2D([0],[0],color=colors[ds],lw=2,label=label) for ds,label in [('kitti','KITTI'),('bonn','Bonn'),('scannetpp','ScanNet++')]]
    legend += [Line2D([0],[0],color='#3b4655',lw=2,ls=ls,label=label) for ls,label in [('-','Ours + TAG'),('--','FT, seed 46'),(':','Released Metric-VDA-S')]]
    ax.grid(alpha=.18);ax.legend(handles=legend,ncol=3,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False)
    fig.savefig(ROOT/'media/static_depth_change.svg')
    fig.savefig(ROOT/'scratch/static_depth_change.png',dpi=150)
    savejson(ROOT/'media/STATIC_CURVES.json',curves)
    print('RENDER FINISHED',flush=True)

if __name__=='__main__':
    for p in ['scratch/tmp','scratch/cache','scratch/mpl','media']: (ROOT/p).mkdir(parents=True,exist_ok=True)
    if sys.argv[1]=='infer': inference()
    elif sys.argv[1]=='render': render()
