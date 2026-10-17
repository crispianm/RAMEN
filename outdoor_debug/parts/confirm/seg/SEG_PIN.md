# SEG_PIN (PLAN ONLY; nothing downloaded; Manager HOLD / Auditor O36, Add.47 provisional; segmenter B swapped to OneFormer per Auditor O37)

## Checkpoints (candidates, to be pinned after the hold lifts)
| role | HF repo | class | licence (model card field) | licence text | revision | weight/config sha256 | size |
|---|---|---|---|---|---|---|---|
| A | nvidia/segformer-b5-finetuned-ade-640-640 | 'sky' expected id 2 (ADE20K-150, 0-based) | `license: other` | card links https://github.com/NVlabs/SegFormer/blob/master/LICENSE = "NVIDIA Source Code License for SegFormer": research/evaluation-only ("non-commercially means for research or evaluation purposes only", s3.3) | to be filled after download | to be filled after download (every file: config.json, preprocessor_config.json, model.safetensors / pytorch_model.bin) | to be filled |
| B | shi-labs/oneformer_cityscapes_swin_large (OneFormer, task token 'semantic') | class named 'sky', Cityscapes trainId 10, asserted BY NAME from id2label | `license: mit` (model card metadata; no separate licence text on the card) | MIT per card; Cityscapes-trained (dataset terms non-commercial; state as a limitation) | to be filled after download | to be filled after download (config.json, preprocessor_config.json, model weights, tokenizer files) | to be filled |
Rejected alternative: facebook/mask2former-swin-large-cityscapes-semantic. Card metadata `license: other` with no licence text or link; Mask2Former GitHub says code 'majority MIT' with portions under separate terms and is silent on the weights: UNRESOLVED (read 2026-10-17, Auditor O37), hence swapped for OneFormer before any use. Not used anywhere.
Licences read via read-only web fetch of model cards/GitHub only; file lists, sizes, shas, commit hashes: sha: to be filled after download. `make_sky_mask.py` requires `--ade-rev/--city-rev` (city default repo = shi-labs/oneformer_cityscapes_swin_large) commit hashes and asserts id2label by name at load.

## Declared thresholds (before any use) - see make_sky_mask.py docstring
prob >= 0.9 in BOTH; working res short side 512 (long side multiple of 32), bilinear prob upsampling to image size, then threshold; boundary band: Euclidean distance to nearest non-certain pixel must be > r, r = round(8*short/900) px (8 px at 1600x900), min 1; LiDAR exclusion r_l = round(5*short/900) px around any projected return (function `lidar_exclusion`/`apply_lidar`, applied by the scorer); drop a frame if < 1% of pixels certain-sky (after erosion and LiDAR); fp32, eval, seed 0. Mask-file: one npz (packed bits per frame id + json meta incl. dropped list), sha256 sidecar.
Differences from the proposal: drop rule is 1% of the image (task spec) instead of >=2000 px at 378 short side (about 0.8%); the proposal's "above the horizon row" and 2-model IoU >= 0.85 agreement check are NOT in the mask function (agreement IoU reportable from the saved probs; ask if wanted).

## To do after the hold lifts
download with huggingface_hub (revision pinned), sha256 all files into SHA256SUMS, fill the table, CPU shape check on a random-noise image only, run make_panel.py on the registered AV2 frames before any prediction.
Env: ~/.venvs/ramen has torch 2.10, numpy, scipy, PIL, safetensors, huggingface_hub but NOT transformers.

## OneFormer score (declared; code: oneformer_semantic_scores in make_sky_mask.py)
Task token: OneFormerProcessor(images=..., task_inputs=["semantic"]) gives the 'the task is semantic' token ids, passed as `task_inputs` to OneFormerForUniversalSegmentation(pixel_values, task_inputs). The processor is used for the task tokens only; pixel_values are our working-res tensor (short side 512, mult of 32, ImageNet mean/std = the processor's own mean/std), so the declared resolution rule is unchanged.
Score = transformers OneFormerImageProcessor.post_process_semantic_segmentation: S[c,h,w] = einsum('bqc,bqhw->bchw', softmax(class_queries_logits,-1)[..., :-1], sigmoid(masks_queries_logits)). Declared extra step: divide by sum_c S so per-pixel class scores sum to 1 (the HF score is an unnormalised argmax score). Computed at the mask-logit resolution (h/4 x w/4), bilinearly upsampled (align_corners=False) to image size BEFORE the >= 0.9 threshold. Open: the task also needs a text tokenizer from the repo (processor loads it); pinned with the other files.

## Environment (not created)
Needed: transformers (OneFormer in core since 4.26; docs read were v5.20.1 -- pin the exact version at creation and record it), torch 2.10 (already in ~/.venvs/ramen), pillow, numpy, scipy, safetensors, huggingface_hub. Tokenizer for the processor: `tokenizers` (pulled by transformers). Mount check (`mount | grep noexec`): /mnt/DATA (/dev/sda1) is noexec, which explains the failed import there (venv .so files cannot be mmap-executed); / (nvme0n1p3, includes ~ and /home/wg19671/.claude/jobs/a0d391cb/tmp) is exec. Recipe: `python3 -m venv ~/.venvs/ramen_seg --system-site-packages` NOT used; instead `python3 -m venv ~/.venvs/ramen_seg && ~/.venvs/ramen_seg/bin/pip install torch==2.10.* pillow numpy scipy safetensors huggingface_hub transformers==<pin> pytest`, keep HF_HOME under ~/.cache/huggingface (exec mount irrelevant for weights but keep downloads in a fresh empty dir), then `pip freeze > seg_env_freeze.txt` and sha it. Weights may stay on /mnt/DATA (data files need no exec).

# PINNED 2026-10-17 (Auditor O40 F1; date -u of download completion ~22:30Z; Add.48 clears registration/data work)
Weights downloaded with huggingface_hub snapshot_download(revision=<commit>) into fresh dirs under /mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/segw/ (data files only; the env is ~/.venvs/ramen_seg2 on the exec root mount: transformers 4.57.1, tokenizers 0.22.2, huggingface_hub 0.36.2, safetensors 0.7.0, torch 2.10.0+cu128, pillow 11.3.0, numpy 1.26.4; torch/numpy/PIL via a .pth to the ramen venv; seg_env_freeze.txt).
Both LFS weight files match the Hub's own LFS sha256 (checked). The large files are the only ones with an independent hash; small files are hashed here only.

| role | repo @ revision (commit) | licence | file | bytes (= Hub listed size) | sha256 |
|---|---|---|---|---|---|
| A SegFormer-B5 ADE20K 640 | nvidia/segformer-b5-finetuned-ade-640-640 @ 739f5d4692954e4a185eac280dec1ba5a7d52f1d | card `license: other` = NVIDIA Source Code License for SegFormer (research/evaluation only) | pytorch_model.bin | 339299397 | a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df |
| A | | | config.json | 6886 | 083fbb6bc7ce4a2fc54bdb08686943cc5a72d14b32e81187b6ae12a58092bc13 |
| A | | | preprocessor_config.json | 271 | 2f6a855d3288caa2a34980aafe7353394fdd5582a5049cb4c9875581854597a8 |
| B OneFormer Cityscapes Swin-L | shi-labs/oneformer_cityscapes_swin_large @ 8ed649db2cc1756b8bc36f0cad00b0812e333027 | card `license: mit` (Cityscapes-trained; dataset terms non-commercial) | pytorch_model.bin | 879382349 | 876855252aa674c7ad6e920c268d6722add811a927806372811faa724f19856c |
| B | | | 250_16_swin_l_oneformer_cityscapes_90k.pth (original ckpt, downloaded by snapshot, NOT loaded) | 949594419 | 5507ff070121255fbc9c356ad18190112fa51f30e8e615d69fa1059279505efd |
| B | | | config.json | 77742 | c6c9e5afa09a495e15b6fe5325f913b9efb68d0cd123c2f96dbd3b8da286bcb3 |
| B | | | preprocessor_config.json | 1525 | bebd71fdf01fe2ec94e4fb7edd5722fa9f5bf0cac8c6f0a382e0594e76a79146 |
| B | | | merges.txt | 524619 | 9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a |
| B | | | vocab.json | 1059962 | e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349 |
| B | | | special_tokens_map.json | 472 | c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac |
| B | | | tokenizer_config.json | 812 | cc030e8362b18814d23cbaa9aa0eda2716d6fdd8c1ed669f82a659408961da13 |
| B extra dependency | dataset shi-labs/oneformer_demo @ 4d683bd5bf84e9c8b5537dce306230bde409fe89 | (dataset card carries no licence field) | cityscapes_panoptic.json (the OneFormer image processor's constructor fetches it; panoptic class metadata, unused by our semantic path) | 838 | e3460e4c6efc7fc80aed97a407a29290aafc2e4c661ae04ee7625a42f08f5614 |

Real-wrapper CPU shape check (random uint8 noise 480x360, no AV2 data; HF_OFFLINE with the pinned cache): loads; sky ids asserted by name: SegFormer ADE20K id 2 'sky', OneFormer Cityscapes id 10 'sky'; sky_prob returns float32 [480,360]; SegFormer range 0.0098-0.528, OneFormer 9e-6-1.4e-5 on noise (noise is not sky; no mask is produced from this); forced-fail: assert_sky_id(... 'sky', 3) raises. The processor warning 'reduce_labels/_max_size ignored' is from the repo's old preprocessor_config (task tokens only are used).
The wrapper loads local dirs; `revision` is then ignored by transformers, so the pin IS the directory content (sha256 above) + the commit hash recorded in the mask file meta.
