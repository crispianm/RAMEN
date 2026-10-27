# configs/paper_results.yaml schema (version 1)

Validated by `repro/config.py`. Unknown keys are errors. Item ids contain ':' (e.g. `tab:static`); output file stem replaces ':' with '_'.

```yaml
schema_version: 1

tag: {alpha: 0.2, tau: 0.06, token: agf0.2t0.06}     # TAG depth post-filter; a model with tag: true gets "+<token>" appended to its spec

host_paths:                                          # optional: where data/ckpts live; ${DATA_ROOT} etc. expanded from env
  DATA_ROOT: /mnt/DATA/DegradedDepth                 # (override by env var of the same name)

models:                                              # registry; items refer to ids
  earlier_s45:
    label: "RAMEN (seed 45)"
    spec: "ours:{ckpt}"                              # bench_joint --model spec. "{ckpt}" is replaced by ckpt.path. Pure-name specs (metric_vda_s, dav2_s, daac, metric_vda_l,
                                                     #   fastdvdnet_blind+metric_vda_s, pocketdvdnet+metric_vda_s) need no ckpt
    tag: false                                       # true -> spec + "+agf0.2t0.06"
    ckpt:                                            # optional for pure-name specs; REQUIRED when spec has {ckpt}
      path: checkpoints_v3/.../step12000.pt          # relative to DATA_ROOT unless absolute
      sha256: 334623ec...                            # full 64 hex (a >=8 hex prefix is accepted with a WARNING)
      download: "https://..."                        # URL or "huggingface:org/repo/file"
    reference_spec: "ours:/projects/b5dh/.../step12000.pt"   # optional: the spec string the RELEASED read ran under (Isambard path); used by --verify to find <ref>/<slug>/bench/... (+TAG appended if tag: true)
    weights: {metric_vda_s: <sha256>}                # optional: sha256 of bench_joint-resolved base weights (hash-checked, recorded in MANIFEST)

datasets:                                            # registry; ids = bench_joint dataset names
  kitti:
    n_clips: 13
    clip_list: provenance/clip_lists/kitti.txt       # optional file, one clip key per line
    clip_list_sha256: ...
    manifest_sha256: 7024ba29...                     # what the result JSON must carry in manifest_sha256 (checked after each unit)
    llvar_draws_sha256: f31055b2...                  # optional, same check
  kitti_minus_0047:
    n_clips: 12
    derive_from: {base: kitti, drop: "2011_10_03_drive_0047_sync"}   # no GPU: derived exactly from the base unit's per-clip JSON

items:
  tab:static:
    kind: bench                                      # bench | night | fnoise | latency | qualitative
    title: "..."
    paper_ref: "tab:static"
    models: [earlier_s45, earlier_s45_tag, ...]          # ids from models; row order = table order
    datasets: [kitti, bonn, scannetpp]
    cells: [st_llvar]                                # bench_joint cells: llvar/llvarf (dynamic), st_llvar/st_llb3 (static), b0s0 (clean), real (night), fnoise
    split: all                                       # dev | test | all.  test/all needs `--allow-test` AND env RAMEN_AUDITOR_TOKEN (or a file named by it); rows labelled "exposed exploratory"
    temporal: true                                   # pass --temporal (TAE, tOF)
    metrics: [abs_rel_aligned, delta1_aligned, flicker, abs_rel_static, tae_gtvalid, tof_native, psnr, ssim, eps_d, joint_p95]   # keys of per-video record / video_macro
    aggregation:
      per_dataset: macro                             # nanmean over clips (== video_macro). only option
      mean_over: [kitti, bonn, scannetpp]            # optional: add a "mean" row = unweighted mean of per-dataset values (subset allowed)
      tae_datasets: [kitti, bonn, scannetpp]         # optional: metrics tae*/tof* are averaged only over these datasets
    seeds: {degradation: "fixed by bench_joint (per dataset/clip/sigma; see llvar_draws_sha256)"}   # informational; recorded in MANIFEST
    reference: results_v3/bfix_s12                   # released reads dir (relative to DATA_ROOT); --verify <ref_dir> overrides. Layout: <ref>/<slug(spec)>/bench/<slug>__<ds>__<cell>__<split>__v3.json
    label: null                                      # optional free text, e.g. "exposed exploratory"
    est_gpu_min_per_unit: {kitti: 6, bonn: 10}       # optional overrides for --dry-run estimate

  tab:latency:
    kind: latency                                    # 4090 only; units = models; shells bench_joint.py latency; metrics ms_per_frame..., flagged >10%
    models: [...]
    host_gpu_contains: "4090"
    ...
  fig:qual:
    kind: qualitative                                # calls repro.qualitative.render_<id with ':'/'-' -> '_'>(cfg_entry: dict, out_dir: Path) -> list[Path]
    models: [...]
    params: {...}                                    # free-form, passed through untouched to the renderer inside cfg_entry
```

Unit = (item, model, dataset, cell, split). Units are de-duplicated across items by key (model_spec, dataset, cell, split, temporal): tab:static and tab:dynamic that need the same run share one execution.
Metrics are resolved from the result JSON `per_video[clip][metric]` (per clip) and `video_macro[metric]`.

## Extensions (schema v1, additive)

```yaml
scorer_commit: 368b5c53...            # pristine commit the scorer is vendored from (default 368b5c53b37e...)
tag_backend: reference                # reference | fast; item-level and CLI --tag-backend override.  fast => env RAMEN_AGF_BACKEND=gpu (the scorer's own device TAG,
                                      #   bit-identical per WP24); repro/tag_fast.py is NOT wired into the scorer (exploratory, unaudited). fast units are planned on the 4090 and carry cross-host tolerances.
vendored: {bench_overrides.py: {sha256, commit}, fidelity_init_score.py: {sha256, commit}}   # informational; the CHECKED copy is repro/vendored/VENDORED.json (sha256 verified before any run)

models:
  restorer_temp2_init:
    spec: "fidelity_init:temp2_distilled"      # label only
    runner: fidelity_init                      # shells out to repro/vendored/fidelity_init_score.py --mode init (restorer-only PSNR/SSIM/tOF; item cells must be [llvar]); needs ckpt.path+sha256
  ft_n47:
    ckpt: {path: /projects/b5dh/..., sha256: ..., download_url: TODO(user)}   # absolute path = Isambard; download_url is the public URL slot (download is the legacy name)

items:
  tab:p1:
    host: "4090"                               # gh200 (default) | "4090" | any. Quote 4090. Smoke and fast-TAG plans move every unit to the 4090; latency defaults to 4090.
    override_script: bench_overrides           # repro/vendored/bench_overrides.py, one unit per arm; row id = <model>@<arm>; wrapper runs in the pristine scorer clone
    arms: {full: {coupling: 1.0, stage2: 1.0}, coup0: {coupling: 0.0, stage2: 1.0}}
  tab:fid:
    seed_mean: [{name: "RAMEN (3-seed mean)", members: [ramen_n46, ramen_n47, ramen_n48]}]   # extra rows per (dataset, cell): unweighted mean of members (+ seed_range min/max); they also enter mean_over rows
    reference: {default: results_v3/syn_s12, by_model: {earlier_s45: results_v3/bfix_s12}}      # per-model released dirs (str still accepted); --verify with no value uses these
```

A unit shared by items planned for different hosts becomes host `any`. `--host auto` (default) detects the GPU and skips units planned for the other host (no GPU detected: runs all).

### Pristine scorer clone
The vendored wrappers (`bench_overrides.py`, `fidelity_init_score.py`) and the figure renderers import `bench_joint` from a checkout and the figure scripts assert HEAD == 368b5c53 and a clean tree. `generate_results.py` therefore creates (once) `git clone --shared` of this repository at `scorer_commit` (default `~/.cache/ramen_scorer/368b5c53`, or `--scorer-clone` / `$RAMEN_SCORER_CLONE`), after checking that this repo's scorer files (`bench_joint.py`, `metrics*.py`, `evaluation/`, the three `scripts/dn01_*` drivers) are byte-identical to that commit; otherwise it refuses. No scorer file is edited or patched. Renderers receive the path as `cfg_entry["_scorer_clone"]`, and `cfg_entry["_tag_backend"]`.

## Freeze commit F additions (Auditor rulings 2026-10-14)
- **Freeze = the commit.** The config header must carry `FROZEN (this commit)` and not the draft marker; a real run refuses otherwise, and refuses a config that is untracked or modified vs HEAD. MANIFEST.json records `frozen`, `git.commit`, `config.sha256`, `data_root`, `run_labels` (per item, per host), `bvi` (clip-list sha 8a093838 + per-file shas of every scored frame, needs `--bvi-root`), `conditional_items`.
- **Paths**: local = env `DATA_ROOT` / `--data-root` > `host_paths.DATA_ROOT`. On a GH200 an explicit per-key map `host_paths.gh200` (config path string -> absolute Isambard path, from isambard_layout/layout_table.tsv) resolves every relative path key; unmapped/null keys are REFUSED (no fallback to local paths), `host_paths.gh200_local_only` lists verify references that stay on the local host; `--data-root`/`DATA_ROOT` disable the map. `host_paths.bvi_root` is the default of `--bvi-root`. Paths `repo:<rel>` resolve inside the clone.
- **Latency items** run the single-session ladder harness (`repro/efficiency/latency/ladder_samples.py` = `run_ladder.py` unmodified + all per-pass samples): `harness: {runner: ladder, stack | stack_from_file, rows: [...], warmup, iters, samples}`; rows are keyed `mvda_bracket` (Metric-VDA-S, upper median of the six bracket runs, all 6 x 30 samples kept), `ramen_raw`, `ramen_tag_v3_<tok>`, `ramen_tag_numpy_<tok>`. Env: `RAMEN_LADDER_PYTHON` (interpreter of the session, default = the running one), `RAMEN_GPU_LOCK` (lock file, default `<out>/GPU_LOCK`). `--verify` reads a released ladder session dir (`rows/<stack>__<id>.json`); >10% = FLAG.
- **`condition: {file}`** makes an item conditional: included iff the file exists (under DATA_ROOT, or `$RAMEN_EFFICIENCY_PRIMARY`); with `harness.stack_from_file` its first non-comment word names the ladder stack. Skipped items are printed and recorded in MANIFEST.
- **`reading:`** pre-declared verdicts computed by `repro/readings.py`: `clean_noninferiority` (verdict column in the item CSV/JSON) and `eps_d_every_dataset` (pairwise statements + disclosure in `<stem>_pairs.csv` and the item JSON).
- A table's units must all be planned for one host (`ConfigError` otherwise); `--tag-backend fast` and `--skip-hash` are refused for real runs (`--skip-hash` is allowed with `--smoke`); every used checkpoint is re-hashed against its pin and a mismatch aborts (no smoke/resume exemption).
- `host_slots: {mac: {label, value: "not measured", authorised: false}}` adds a reserved MacBook column to a latency table; no mac executor exists. `previous_host` (informational, planner delta), `verify_class: cross_host` (tab:p1).
