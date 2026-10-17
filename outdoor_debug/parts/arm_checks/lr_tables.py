"""LR tables for the three continuation schedules via the REAL train.make_lr_scale and the C3 resolved config. Run from repo root."""
import sys, yaml, torch
sys.path.insert(0, ".")
import train
BASE = "outdoor_debug/parts/C2_config_train/effective_c3_seed45.yaml"
cfg = yaml.safe_load(open(BASE)); t0 = cfg["training"]
CKPT = sys.argv[1] if len(sys.argv) > 1 else None
groups = [("group0 (all non-gate)", float(t0["learning_rate"])), ("group1 (gates g,g2: x gate_lr_scale)", float(t0["learning_rate"]) * float(t0["gate_lr_scale"]))]
if CKPT:   # actual optimizer groups from the checkpoint
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("checkpoint optimizer param_groups (lr, initial_lr, n_params):", [(g["lr"], g.get("initial_lr"), len(g["params"])) for g in ck["optimizer"]["param_groups"]])
    groups = [(f"ckpt group{i}", g["initial_lr"]) for i, g in enumerate(ck["optimizer"]["param_groups"])]
print("base LRs:", groups)
SCHED = {"S0 (as-is C3 yaml, max_steps 14000)": {"lr_decay_steps": 12000},
         "S1 cosine tail": {"lr_decay_steps": 14000},
         "S2 constant": {"lr_decay_steps": 10**9},
         "S3 restart peak 0.5": {"lr_decay_steps": 14000, "lr_restart_step": 12000, "lr_restart_warmup": 100, "lr_restart_peak": 0.5},
         "S3 restart peak 1.0 (Add.45, SELECTED)": {"lr_decay_steps": 14000, "lr_restart_step": 12000, "lr_restart_warmup": 100, "lr_restart_peak": 1.0}}
steps = [12000, 12001, 12050, 12100, 13000, 13050, 13999]
print("| schedule | " + " | ".join(f"s{s}" for s in steps) + " |")
for n, o in SCHED.items():
    t = dict(t0, max_steps=14000, **o)
    f = train.make_lr_scale(t, int(t["warmup_steps"]), 14000)
    print(f"| {n} scale | " + " | ".join(f"{f(s):.6g}" for s in steps) + " |")
    for gn, b in groups:
        print(f"|   {gn} LR | " + " | ".join(f"{b*f(s):.4g}" for s in steps) + " |")
