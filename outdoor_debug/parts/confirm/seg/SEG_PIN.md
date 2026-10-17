# SEG_PIN (PLAN ONLY; nothing downloaded; Manager HOLD / Auditor O36, Add.47 provisional)

## Checkpoints (candidates, to be pinned after the hold lifts)
| role | HF repo | class | licence (model card field) | licence text | revision | weight/config sha256 | size |
|---|---|---|---|---|---|---|---|
| A | nvidia/segformer-b5-finetuned-ade-640-640 | 'sky' expected id 2 (ADE20K-150, 0-based) | `license: other` | card links https://github.com/NVlabs/SegFormer/blob/master/LICENSE = "NVIDIA Source Code License for SegFormer": research/evaluation-only ("non-commercially means for research or evaluation purposes only", s3.3) | to be filled after download | to be filled after download (every file: config.json, preprocessor_config.json, model.safetensors / pytorch_model.bin) | to be filled |
| B | facebook/mask2former-swin-large-cityscapes-semantic | 'sky' expected trainId 10 | `license: other` | repo code licence is MIT (Meta, 2022, facebookresearch/Mask2Former LICENSE); weights terms NOT verified from the card text: UNVERIFIED, read the card/README at pin time | to be filled after download | to be filled | to be filled (~0.2B params, F32) |
Alternative for B if its licence is unsuitable: shi-labs/oneformer_cityscapes_swin_large (licence UNVERIFIED). Note: Mask2Former Cityscapes weights are trained on Cityscapes (non-commercial dataset terms); state as a limitation.
Licences read via read-only web fetch of model cards/GitHub only; file lists, sizes, shas, commit hashes: sha: to be filled after download. `make_sky_mask.py` requires `--ade-rev/--city-rev` commit hashes and asserts id2label by name at load.

## Declared thresholds (before any use) - see make_sky_mask.py docstring
prob >= 0.9 in BOTH; working res short side 512 (long side multiple of 32), bilinear prob upsampling to image size, then threshold; boundary band: Euclidean distance to nearest non-certain pixel must be > r, r = round(8*short/900) px (8 px at 1600x900), min 1; LiDAR exclusion r_l = round(5*short/900) px around any projected return (function `lidar_exclusion`/`apply_lidar`, applied by the scorer); drop a frame if < 1% of pixels certain-sky (after erosion and LiDAR); fp32, eval, seed 0. Mask-file: one npz (packed bits per frame id + json meta incl. dropped list), sha256 sidecar.
Differences from the proposal: drop rule is 1% of the image (task spec) instead of >=2000 px at 378 short side (about 0.8%); the proposal's "above the horizon row" and 2-model IoU >= 0.85 agreement check are NOT in the mask function (agreement IoU reportable from the saved probs; ask if wanted).

## To do after the hold lifts
download with huggingface_hub (revision pinned), sha256 all files into SHA256SUMS, fill the table, CPU shape check on a random-noise image only, run make_panel.py on the registered AV2 frames before any prediction.
Env: ~/.venvs/ramen has torch 2.10, numpy, scipy, PIL, safetensors, huggingface_hub but NOT transformers.
