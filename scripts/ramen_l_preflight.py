#!/usr/bin/env python3
"""RAMEN-L submit preflight (experimental ViT-L variant). Reads files and the git state, optionally builds the model on CPU.
Exit 2 and `REFUSED: ...` lines on any failure; `PREFLIGHT OK` otherwise. Every expected value below is a literal in THIS file.

  python3 -I scripts/ramen_l_preflight.py --config configs/experiments/ramen_l_c3_n46.yaml [--smoke | --resume] --pin <40hex> \
      --out-dir runs/ramen_l_c3_n46 [--weights /path/to/metric_video_depth_anything_vitl.pth] [--build] [--no-git]

Checks: (1) HEAD == pin and a clean tree; (2) the released Metric-VDA-L file: size + sha256; (3) the RESOLVED config equals the resolved
Tables 1/2 config (dn01_c3rep_20261009_c3_n46) except for an allow-listed set of keys (nothing else may differ: data mix, seed, loss, LR, batch);
(4) the config pins the same weights sha; (5) every path the config names exists; (6) the output dir is absent; (7) --build: the ViT-L model
builds from the file (strict load), embed_dim 1024 / 24 blocks / DPT width 256, trainable parameter count == the literal for the chosen restorer.
--resume (second segment of a run that hit the wall limit; config = <full config stem>_resume.yaml or --full-config): the out dir MUST exist and hold last.pt; the only
config difference from the full config is training.resume_from == <checkpoint_dir>/last.pt (relative, inside the out dir); last.pt must be a resumable checkpoint
(optimizer, rng, ema, trainable, cfg, step, micro), 500 <= step <= 11500 and a multiple of 500, with ViT-L-shaped tensors, and its saved cfg must resolve to the
SAME config as the committed full config (both segments are one recipe)."""
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_CFG = "configs/experiments/dn01_c3rep_20261009_c3_n46.yaml"
BASE_RESOLVED_SHA = "46c78d52a20828f00ec79071a93a38bc8c4ff734458f6890e350bb5f3163d56c"   # sha256 of json.dumps(load_config(BASE_CFG), sort_keys=True)
WEIGHTS_SHA = "24eba25342e7ee0f054be25da6a852ebd7cfa40cd6bf41a353ecb6abfe24e620"
WEIGHTS_SIZE = 1538348216
ALLOWED_DIFF = {"model.vda_encoder", "model.vda_checkpoint_path", "model.vda_checkpoint_sha256", "model.vda_encoder_grad_ckpt",
                "model.st_stage2_width", "training.checkpoint_dir"}
SMOKE_DIFF = {"training.max_steps", "training.lr_decay_steps", "training.checkpoint_at_steps"}
WANT = {"model.vda_encoder": "vitl", "model.vda_init": "released", "model.metric": True, "model.shared_temporal": True, "model.st_depth_input": "restored",
        "data.seed": 46, "training.warmup_steps": 500, "training.batch_size": 1, "training.grad_accum": 4, "training.learning_rate": 0.0001,
        "model.vda_encoder_grad_ckpt": True}
STAGE2 = {"wide2x", "wide3x", "wide4x"}
# trainable parameters per restorer width: DPT head (incl. 49.07 M motion modules) 80,021,697 + temp2 321,618 + block2 + two scalar gates
TRAINABLE = {"wide2x": 80021697 + 321618 + 1248882 + 2, "wide3x": 80021697 + 321618 + 2784402 + 2, "wide4x": 80021697 + 321618 + 4928178 + 2}


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def _rel_inside(p, d):
    """True if p is a relative path of the form <d>/<name> without parent references."""
    return isinstance(p, str) and not p.startswith(("/", "~", "$")) and ".." not in Path(p).parts and Path(p).parent == Path(d)


def check_resume(a, cfg, fc, R, train):
    """--resume: every refusal is appended through R. Returns a short note for the OK line."""
    out = a.out_dir
    if not out:
        R("--resume needs --out-dir"); return ""
    rf = fc.get("training.resume_from")
    if rf != f"{out}/last.pt" or not _rel_inside(rf, out):
        R(f"training.resume_from {rf!r} must be the relative path '{out}/last.pt' inside the out dir")
    full_path = Path(a.full_config) if a.full_config else Path(re.sub(r"_resume(\.ya?ml)$", r"\1", a.config))
    if str(full_path) == str(a.config):
        R("resume config must be named <full config>_resume.yaml or --full-config must be given"); return ""
    full = train.load_config(full_path)
    ff = flat(full)
    d = {k for k in set(fc) | set(ff) if fc.get(k) != ff.get(k)}
    if d - {"training.resume_from"}:
        R("resume config differs from the full config beyond training.resume_from: " + ", ".join(sorted(d - {"training.resume_from"})))
    if ff.get("training.resume_from"):
        R("the full config itself carries training.resume_from")
    od = REPO / out
    last = REPO / (rf or "")
    if not od.is_dir():
        R(f"--resume: out dir {out} does not exist"); return ""
    if not last.is_file():
        R(f"--resume: {out}/last.pt does not exist"); return ""
    import torch
    try:
        ck = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as e:
        R(f"last.pt unreadable: {type(e).__name__}: {str(e)[:100]}"); return ""
    miss = [k for k in ("optimizer", "rng", "ema", "step", "micro", "cfg") if k not in ck] + ([] if ("trainable" in ck or "trainable_ema" in ck) else ["trainable"])
    if miss:
        R(f"last.pt is not a resumable checkpoint (train.py needs optimizer and rng): missing {miss}"); return ""
    step = int(ck["step"])
    if not a.allow_any_step and not (500 <= step <= 12000 - 500 and step % 500 == 0):
        R(f"last.pt step {step}: must be a multiple of 500 in [500, 11500] (12000 = already finished)")
    if a.expect_step is not None and step != a.expect_step:
        R(f"last.pt step {step} != --expect-step {a.expect_step}")
    sj = lambda d_: json.dumps(d_, sort_keys=True)
    if sj(ck["cfg"]) not in (sj(full), sj(cfg)):
        bk = ck["cfg"]; dk = sorted(k for k in set(flat(bk)) | set(ff) if flat(bk).get(k) != ff.get(k))
        R(f"last.pt saved cfg does not resolve to the committed full config (nor to this resume config); differing keys {dk[:6]}")
    tr = ck.get("trainable_ema", ck.get("trainable")) or {}
    p0 = next((v for k, v in tr.items() if k.endswith("projects.0.weight")), None)
    if p0 is None or list(p0.shape) != [256, 1024, 1, 1]:
        R(f"last.pt trainable tensors are not ViT-L shaped (DPT projects.0 {None if p0 is None else list(p0.shape)})")
    return f" resume_step={step}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--full-config", help="resume mode: the committed full config the checkpoint was trained with (default: --config with a trailing _resume removed)"); ap.add_argument("--expect-step", type=int, help="resume mode: last.pt must be at exactly this step")
    ap.add_argument("--allow-any-step", action="store_true", help="CONTROL ONLY: skip the step range / multiple-of-500 rule"); ap.add_argument("--pin"); ap.add_argument("--out-dir")
    ap.add_argument("--weights"); ap.add_argument("--build", action="store_true"); ap.add_argument("--no-git", action="store_true", help="CONTROL ONLY")
    ap.add_argument("--no-paths", action="store_true", help="CONTROL ONLY: skip the data-path existence checks"); ap.add_argument("--weights-size", type=int, default=WEIGHTS_SIZE, help="CONTROL ONLY")
    ap.add_argument("--base-resolved-sha", default=BASE_RESOLVED_SHA, help="CONTROL ONLY"); ap.add_argument("--weights-sha", default=WEIGHTS_SHA, help="CONTROL ONLY")
    a = ap.parse_args(argv)
    bad = []
    R = bad.append
    if not a.no_git:
        if not (a.pin and re.fullmatch(r"[0-9a-f]{40}", a.pin)):
            R(f"--pin {a.pin!r} is not a 40-hex sha")
        try:
            head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
            if a.pin and head != a.pin:
                R(f"HEAD {head} != pin {a.pin}")
            por = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
            if por.strip():
                R("tree dirty: " + por.strip().replace("\n", " | ")[:200])
        except Exception as e:
            R(f"git failed: {e}")
    wpath = Path(a.weights or os.path.expandvars("${RAMEN_WEIGHTS}/metric_video_depth_anything_vitl.pth"))
    if not wpath.is_file():
        R(f"weights file missing: {wpath}")
    else:
        if wpath.stat().st_size != a.weights_size:
            R(f"weights size {wpath.stat().st_size} != {a.weights_size}")
        g = sha256(wpath)
        if g != a.weights_sha:
            R(f"weights sha256 {g} != pinned {a.weights_sha}")
    sys.path.insert(0, str(REPO)); sys.dont_write_bytecode = True
    import train
    cfg = train.load_config(Path(a.config))
    base = train.load_config(REPO / BASE_CFG)
    bsha = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    if bsha != a.base_resolved_sha:
        R(f"resolved Tables 1/2 config sha {bsha} != literal {a.base_resolved_sha} (the baseline config drifted)")
    fc, fb = flat(cfg), flat(base)
    diff = {k for k in set(fc) | set(fb) if fc.get(k) != fb.get(k)}
    if a.resume and a.smoke:
        R("--resume and --smoke are exclusive")
    allowed = ALLOWED_DIFF | (SMOKE_DIFF if a.smoke else set()) | ({"training.resume_from"} if a.resume else set())
    if diff - allowed:
        R("keys differ from the Tables 1/2 recipe outside the allow-list: " + ", ".join(f"{k}: {fb.get(k)!r} -> {fc.get(k)!r}" for k in sorted(diff - allowed)))
    if not a.smoke and (SMOKE_DIFF & diff):
        R("a full run may not change " + ", ".join(sorted(SMOKE_DIFF & diff)))
    for k, v in WANT.items():
        if fc.get(k) != v:
            R(f"config {k} = {fc.get(k)!r}, want {v!r}")
    if fc.get("model.st_stage2_width") not in STAGE2:
        R(f"model.st_stage2_width {fc.get('model.st_stage2_width')!r} not in {sorted(STAGE2)}")
    if fc.get("model.vda_checkpoint_sha256") != a.weights_sha:
        R(f"config model.vda_checkpoint_sha256 {fc.get('model.vda_checkpoint_sha256')!r} != the preflight literal")
    if a.smoke:
        if fc.get("training.max_steps") != fc.get("training.lr_decay_steps"):
            R("smoke: lr_decay_steps != max_steps")
    elif (fc.get("training.max_steps"), fc.get("training.lr_decay_steps")) != (12000, 12000):
        R("full run: max_steps / lr_decay_steps are not 12000")
    ck = fc.get("training.checkpoint_dir", "")
    if a.out_dir and ck != a.out_dir:
        R(f"config training.checkpoint_dir {ck!r} != --out-dir {a.out_dir!r}")
    for k, v in ({} if a.no_paths else fc).items():
        if re.search(r"(^|\.)(root|gt_root|extra_root|vda_repo|st_temp2_init)$", k) and isinstance(v, str):
            if not Path(os.path.expandvars(v)).exists():
                R(f"path named by {k} does not exist: {v}")
    for i, s in enumerate([] if a.no_paths else cfg.get("data", {}).get("sources", [])):
        for k in ("root", "gt_root", "extra_root"):
            if isinstance(s.get(k), str) and not Path(s[k]).exists():
                R(f"data.sources[{i}].{k} does not exist: {s[k]}")
    resume_note = ""
    if a.resume:
        resume_note = check_resume(a, cfg, fc, R, train)
    elif a.out_dir and (REPO / a.out_dir).exists():
        R(f"output dir {a.out_dir} already exists")
    elif fc.get("training.resume_from"):
        R("training.resume_from is set but --resume was not given")
    n_params = None
    if a.build and not bad:
        import torch
        from models import build_model, trainable_modules
        bcfg = json.loads(json.dumps(cfg))
        if not Path(os.path.expandvars(str(bcfg["model"].get("st_temp2_init") or ""))).is_file():   # values are irrelevant to the dims / parameter count
            bcfg["model"]["st_temp2_init"] = None
        bcfg["model"]["vda_checkpoint_path"] = str(wpath)             # the file whose sha256 was just verified
        m = build_model(bcfg)
        c = m.vda.core
        assert (c.pretrained.embed_dim, len(c.pretrained.blocks), c.head.scratch.output_conv1.in_channels) == (1024, 24, 256), "not a ViT-L model"
        n_params = sum(p.numel() for p in m.trainable_parameters())
        want = TRAINABLE[fc["model.st_stage2_width"]]
        if n_params != want:
            R(f"trainable parameters {n_params} != literal {want} for {fc['model.st_stage2_width']}")
        if any(p.requires_grad for p in c.pretrained.parameters()):
            R("encoder is not frozen")
    if bad:
        for b in bad:
            print("REFUSED: " + b)
        return 2
    print(f"PREFLIGHT OK (ramen-l) config={a.config} smoke={a.smoke} resume={a.resume}{resume_note} stage2={fc['model.st_stage2_width']} weights_sha={a.weights_sha[:12]} trainable_params={n_params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
