"""R-c: what fixes post-resume data order / noise through the REAL code path (train._build_multisource, train.seed_everything, train._restore_rng_state, train._start_state, real datasets, local roots).
scannetpp_train is dropped (no local split list), num_workers 0. Fake resume rng = the state seed_everything(777) + 5 draws would leave (same for every arm). Run from repo root."""
import sys, copy, json
from itertools import islice
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))
import torch, yaml, train
from arms import build, apply
from classes import remap
N = 200

def make(cfg, world):
    c = remap(cfg)
    c["data"]["sources"] = [s for s in c["data"]["sources"] if s["type"] != "scannetpp_train"]
    c["data"]["val_sources"] = [s for s in c["data"].get("val_sources", []) if s["type"] != "scannetpp_train"]
    c["data"]["num_workers"] = 0
    t = c["training"]; d = c["data"]
    train.seed_everything(int(d["seed"]), 0, str(t.get("seed_scheme", "add")))       # train.py main(): before the loaders are built
    loader, sampler = train._build_multisource(c, "train", world, 0)                  # builds the sampler (+ its own generator when world>1)
    for k in ("clean_prob", "seeded_noise"):                                          # the attrs train.py sets after the build
        train._set_dataset_attr(loader.dataset, k, d.get(k, False if k == "seeded_noise" else 0.0))
    for k in ("lowlight", "static_prob", "blur_only_prob", "blur_disk_prob", "static_exempt_shares", "static_lowlight_only"):
        if k in d: train._set_dataset_attr(loader.dataset, k, d[k])
    # fake resume: every arm restores the SAME checkpoint rng state
    train.seed_everything(777, 0, "x1000"); [torch.rand(1) for _ in range(5)]
    st = train._rng_state(); train.seed_everything(1, 0, "add")                       # scramble, then restore like train.py:2676
    train._restore_rng_state(st)
    train._start_state(loader.dataset, t, 12000, 0, True)                             # resume_reseed applies here
    return c, loader, sampler

def cid(ds, i):
    from torch.utils.data import ConcatDataset
    k = 0
    while i >= ds.cumulative_sizes[k]: k += 1
    j = i - (ds.cumulative_sizes[k - 1] if k else 0); p = ds.datasets[k]
    cl = p.clips[j]
    return (k, str(Path(str(cl[0][0])).parent.parent.name) + "/" + Path(str(cl[0][0])).name, cl[3])

def run(name, cfg, world, fixed_idx=None):
    c, loader, sampler = make(cfg, world)
    idx = list(islice(iter(sampler), N))
    ds = loader.dataset
    clips = [cid(ds, i) for i in idx]
    base_seed = int(torch.empty((), dtype=torch.int64).random_().item())   # what a DataLoader draws from the global RNG for its worker seeds (iterator creation, after the sampler iter above consumed its draw if gen None)
    return c, loader, idx, clips

if __name__ == "__main__":
    base, arms = build()
    A0 = arms["A0"]
    variants = {"A0": A0, "TS2": arms["TS2"],
                "A0b[data.seed46]": apply(A0, [("data.seed", 46)]),
                "A0b[resume_reseed46]": apply(A0, [("training.resume_reseed", 46)]),
                "A0b[both]": apply(A0, [("data.seed", 46), ("training.resume_reseed", 46)])}
    out = {}
    for world in (4, 1):
        res = {}
        for n, cfg in variants.items():
            c, loader, idx, clips = run(n, cfg, world)
            # noise: fetch the SAME 2 A0 indices (fixed) with this variant's post-_start_state global RNG
            res[n] = dict(idx=idx, clips=clips, loader=loader, rngstate=train._rng_state())
        a0 = res["A0"]
        fixed = a0["idx"][:2]
        print(f"\n=== world_size {world}: first {N} sampler indices / clip ids vs A0 ===")
        for n, r in res.items():
            same_idx = sum(x == y for x, y in zip(r["idx"], a0["idx"])); same_clip = sum(x == y for x, y in zip(r["clips"], a0["clips"]))
            print(f"{n:24s} identical idx {same_idx}/{N}  identical clip ids {same_clip}/{N}")
        # noise draws: actual dataset __getitem__ (real degrade path, seeded_noise) of the same 2 clips under each variant's own post-start global RNG state
        fr = {}
        for n, r in res.items():
            train._restore_rng_state(r["rngstate"])   # state left by _start_state for that variant
            fr[n] = [r["loader"].dataset[i]["frames"] for i in fixed]
        print(f"noise draws (frames of the same {len(fixed)} clips, real __getitem__, seeded_noise) equal to A0's:")
        for n in fr:
            print(f"{n:24s}", [bool(torch.equal(a, b)) for a, b in zip(fr[n], fr['A0'])])
        out[world] = {n: dict(same_idx=sum(x == y for x, y in zip(r["idx"], a0["idx"])), same_clip=sum(x == y for x, y in zip(r["clips"], a0["clips"]))) for n, r in res.items()}
    json.dump({str(k): v for k, v in out.items()}, open(Path(__file__).parent / "seed_check.json", "w"), indent=1)
