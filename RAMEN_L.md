# RAMEN-L (experimental)

RAMEN-L is the Tables 1/2 RAMEN recipe built on the released **Metric-Video-Depth-Anything-Large** (ViT-L) instead of the Small
model. It is an experiment: no result from it is part of the paper and this branch is not a release.

## What differs from RAMEN-S

* Encoder and initialisation: the released Metric-VDA-L (frozen DINOv2 ViT-L; the DPT head and motion modules start from it).
* The gated second restorer block can be scaled up (`model.st_stage2_width`: `wide2x` = unchanged, `wide3x`, `wide4x`).
* `model.vda_encoder_grad_ckpt: true` recomputes the frozen trunk in the backward pass (numerically identical; needed for memory).
* `scripts/ramen_l_preflight.py` refuses a run whose resolved config differs from the Tables 1/2 config
  (`dn01_c3rep_20261009_c3_n46`) anywhere except the allow-listed keys above.

Everything else (data mix, seed 46, 12000 steps, loss weights, LR schedule, batch 1 x accum 4 x 4 GPUs) is inherited unchanged.
At step 0 the model reproduces the released Metric-VDA-L depth bit for bit (the restorer gates start at zero).

## Weights (not part of this repository)

Download `metric_video_depth_anything_vitl.pth` from
`https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth`
into a directory and `export RAMEN_WEIGHTS=<that directory>`.

* size 1538348216 bytes
* sha256 `24eba25342e7ee0f054be25da6a852ebd7cfa40cd6bf41a353ecb6abfe24e620` (enforced by the config and by the preflight)

**Licence.** Video-Depth-Anything-Large is released under CC-BY-NC-4.0 (non-commercial; the Small model is Apache-2.0). A RAMEN-L
checkpoint derived from it inherits that restriction.

## Run

```
export RAMEN_WEIGHTS=...                       # directory with the file above
python scripts/ramen_l_preflight.py --config configs/experiments/ramen_l_c3_n46.yaml --pin $(git rev-parse HEAD) --out-dir runs/ramen_l_c3_n46 --build
torchrun --standalone --nproc_per_node=4 train.py --config configs/experiments/ramen_l_c3_n46.yaml
```

`jobs/ramen_l_train.sbatch` wraps both steps for Slurm (all site-specific values come from environment variables).
Configs: `ramen_l_c3_n46.yaml` (wide4x), `ramen_l_c3_n46_r3x.yaml` (wide3x), `ramen_l_c3_n46_r2x.yaml` (restorer unchanged),
`ramen_l_smoke.yaml` (+ `_r3x`, `_r2x`; 120 steps, memory and speed only).
