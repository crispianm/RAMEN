# RAMEN: Stable Metric Video Depth in Low Light

Code to train and evaluate **RAMEN**, the model of the CVPR 2027 submission *"RAMEN: Stable Metric Video Depth in Low Light"* (Crispian Morris, Fan Zhang, David Bull, Nantheera Anantrasirichai).

RAMEN starts from Metric Video Depth Anything (Small). A small video restorer is trained jointly with the depth head on degraded clips (noise, blur, low light) while the image encoder stays frozen; the restorer supplies the depth network's input and a denoised video output. At inference, an innovation-adaptive exponential smoother (the TAG filter, `evaluation/depth_probe.py`, token `agf0.2t0.06`) suppresses small changes in predicted metric depth and raises its gain when changes are large. This repository is the reproducibility release: the headline architecture, its training recipe, the matched depth-only fine-tuning baseline, the degradation pipeline, the scorer, and the single-command results generator. Research code, experiment configs and run logs are not part of this release.

## Pretrained models

| Model | File | Licence |
|---|---|---|
| RAMEN | `ramen.pt` | Apache-2.0 |
| RAMEN-L | `ramen_l.pt` | CC-BY-NC-4.0 (inherited from Video-Depth-Anything-Large; non-commercial) |

Checkpoints will be released with the paper (see [CHECKPOINTS.md](CHECKPOINTS.md) for the file list and the sha256 pins used by the evaluation). Download links: TODO (none exist yet). Both files hold the depth path and the display head (the denoised video output). **Load them with `models.load_ramen(path)`**; `load_d3_checkpoint` and the `ours:` scorer spec refuse such a file (they would silently drop the display head), and the scorer spec for them is `l15d:<path>`. Offline: `python infer.py --ckpt checkpoints_release/ramen.pt --camera clip.mp4 --save-rgb out/` writes depth and display-head RGB. Streaming mode is not yet supported for these files (see the known issue below); clip mode reproduces the tables.

**Known issue.** Streaming mode (`infer.py` without `--save-rgb`, and `--selftest`) is not yet supported for `ramen.pt` / `ramen_l.pt`; it exits early with a message saying so. The clip mode reproduces the tables.

**Running `infer.py`.** The checkpoint is resolved as `--ckpt` > `RAMEN_CKPT` env > `<repo>/checkpoints_release/ramen.pt`. If the file is missing, the script prints where to put it (download links: TODO, none exist yet) instead of a stack trace.

```
python infer.py --camera clip.mp4 --save-rgb out/                                   # default: <repo>/checkpoints_release/ramen.pt
python infer.py --ckpt /path/to/ramen.pt --camera clip.mp4 --save-rgb out/          # explicit
RAMEN_CKPT=/path/to/ramen.pt python infer.py --camera clip.mp4 --save-rgb out/      # environment
python infer.py --camera clip.mp4 --save-rgb out/ --ckpt checkpoints_release/ramen_l.pt   # RAMEN-L
```

## Installation

```bash
git clone https://github.com/crispianm/RAMEN.git && cd RAMEN
uv sync            # Python >= 3.11, locked environment (uv.lock)
uv run python -c "import models, train, infer"
```

RAMEN builds on code and weights from other projects that are **not** vendored here; clone / download them and point the config at them:

| What | Used for | Config keys |
|---|---|---|
| [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) (code) | spatio-temporal depth head | `model.vda_repo` |
| `metric_video_depth_anything_vits.pth` (Metric-VDA-S) | initialisation of RAMEN and of the fine-tuning baseline | `model.vda_checkpoint_path` |
| `depth_anything_v2_vits.pth` (Depth Anything V2-S) | frozen encoder weights | `model.da_v2_checkpoint_path` |
| `model.st_temp2_init` (distilled restorer stage, see `scripts/dn01_st_distill.py`) | RAMEN only: initialisation of the restorer | `model.st_temp2_init` |

The training configs in `configs/experiments/` are the exact configs of the paper runs; their `root:` / `*_repo` / `*_path` entries are the cluster paths used there. Edit them (or copy the file) to your own locations before training.

## Data

Nothing is downloaded automatically. Obtain each dataset from its official source and prepare it as below.

Training mix (`data.sources` in `configs/experiments/dn01_c3rep_20261009_c3_n46.yaml`): TartanAir, Virtual KITTI 2, PointOdyssey (train and val, filtered by `provenance/dn01_vdamix/po_filter_v1.json`), IRS, Spring, KITTI raw with the depth annotations (drive split: `provenance/dn01_v3/kitti_raw_train_drives.json`, built by `scripts/dn01_v3_kitti_raw_manifest.py`), NYUv2 (train), ScanNet++ (train subset: `provenance/dn01_v3/scannetpp_train_subset140.txt`). Degradations (noise, blur, low-light) are applied on the fly by `data/`; their calibration is in `provenance/dn01_lowlight_calibration/` and `provenance/dn01_gain_calibration/`.

Evaluation sets (clip lists, sha256-pinned in `configs/paper_results.yaml`, section `datasets:`):

| Set | Preparation |
|---|---|
| KITTI (13 drives) | `scripts/dn01_kitti_rgbfix.py` writes the channel-corrected frame root; manifest from `scripts/build_kitti_recipe_manifest.py` |
| Bonn RGB-D | `scripts/build_bonn_recipe_manifest.py` (long-video clip: `repro/qualitative/long_video/build_bonn_long_manifest.py`) |
| ScanNet++ (62 scenes) | `scripts/dn01_v3_scannetpp_eval_export.py` |
| NYUv2 (video clips) and Sintel | clip lists named in `configs/paper_results.yaml` (`nyuv2`, `sintel_rgbfix`) |
| nuScenes night | clip list `provenance/dn01_nuscenes_night_test/nuscenes_night_test_v1.json` (frames from nuScenes) |
| BVI-RLV (real low-light pairs) | pass the frame root with `--bvi-root`; scene registration in `provenance/dn01_fnoise/registration.json` |

Set `DATA_ROOT` (or pass `--data-root`) so that the relative paths in `configs/paper_results.yaml` resolve.

## Training

The headline model and the matched depth-only fine-tuning baseline share data, schedule and seeds; they differ in the model block. Fresh-start runs (12,000 steps, seeds 46, 47, 48):

```bash
# RAMEN
uv run torchrun --standalone --nproc_per_node=4 train.py --config configs/experiments/dn01_c3rep_20261009_c3_n46.yaml
# matched depth-only fine-tune of Metric-VDA-S
uv run torchrun --standalone --nproc_per_node=4 train.py --config configs/experiments/dn01_c3rep_20261009_ft_n46.yaml
```

(`..._n47` and `..._n48` for the other seeds; per-frame DAv2-S / DA-AC baselines: `dn01_c3rep_pf_dav2a_n46.yaml`, `dn01_c3rep_pf_daacm_n46.yaml`.) The paper runs used 4 GPUs, batch 1 and `training.grad_accum: 4`; on one GPU use `python train.py --config ...` and set `training.grad_accum: 16`. Under Slurm, `train.py` requires `VAL01_EXPECTED_COMMIT=$(git rev-parse HEAD)` and a clean tree; see `jobs/example_slurm.sh`.

The earlier seed-45 checkpoint (a continuation: steps 0-6,000 `configs/experiments/dn01_v3_synreal_r50_B.yaml`, steps 6,000-12,000 `configs/experiments/dn01_v3_bfix_C3_B.yaml`) is an earlier checkpoint, not RAMEN, and is not released (no download); its config id is `earlier_s45`. `ramen.pt` uses the seed-46 depth path with a display head fine-tuned for 1500 steps from S3000 (restorer only; rebalanced data plus an edge loss); the display head was adopted by the authors.

## Evaluation (paper tables and figures)

Every table and figure is produced by one command from one frozen config, `configs/paper_results.yaml` (schema in `repro/SCHEMA.md`):

```bash
uv run python generate_results.py --dry-run                 # plan and GPU-hour estimate, runs nothing
uv run python generate_results.py --smoke --dry-run         # the smoke plan
uv run python generate_results.py --data-root $DATA_ROOT     # all items -> paper_results/
uv run python generate_results.py --only tab:static fig:qual
uv run python generate_results.py --verify <reference_dir>   # compare with a previous run
```

Interrupted runs resume. The run refuses a dirty git tree (`--allow-dirty` is recorded in `MANIFEST.json`). Units that read the TEST clips are labelled "exposed exploratory" in the output and need `--allow-test` plus a non-empty `RAMEN_AUDITOR_TOKEN` environment variable (an audit gate: any non-empty value passes). Latency items must be run on the stated GPU. Output layout: `<item>.csv/.json`, `per_clip/` (raw scorer JSON), `qualitative/`, `MANIFEST.json`.

Items that call the vendored wrapper scripts (`repro/vendored/`) and the figure renderers run in a pristine checkout of scorer commit `368b5c53b37ebd22f5d44ce0c9d514122efc88da`, which is not in this repository's single-commit history. To run them, restore it from the archive described in [PROVENANCE.md](PROVENANCE.md) (`git fetch <archive bundle> 'refs/heads/*:refs/archive/*'`); all other items do not need it.

The scorer is `bench_joint.py`, `metrics.py`, `metrics_temporal.py` and `evaluation/`.

## Tests

```bash
uv run python -m pytest tests          # CPU only
```

## License

Code: Apache-2.0 ([LICENSE](LICENSE)), including the vendored PocketDVDNet file (Apache-2.0, relicensed by its creator; see [NOTICE](NOTICE)). Vendored third-party code keeps its own licence (see [NOTICE](NOTICE), which lists every vendored directory).

Weights, as for Video Depth Anything: the RAMEN model (`ramen.pt`, built on the Small model) is under the Apache-2.0 license; the RAMEN-L model (`ramen_l.pt`, built on Video-Depth-Anything-Large) is under the CC-BY-NC-4.0 license (non-commercial). Pretrained weights of other projects that this code loads are subject to their own licences. The Metric-VDA-L fine-tuned baseline and the earlier seed-45 checkpoint are not released.

## Citation

```bibtex
@inproceedings{morris2027ramen,
  title     = {RAMEN: Stable Metric Video Depth in Low Light},
  author    = {Morris, Crispian and Zhang, Fan and Bull, David and Anantrasirichai, Nantheera},
  booktitle = {TODO: venue (CVPR 2027) and pages},
  year      = {2027}
}
```
