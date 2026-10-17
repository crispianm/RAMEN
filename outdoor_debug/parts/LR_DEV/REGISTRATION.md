# Long-range LiDAR DEV set: DDAD val (report-only), REGISTRATION
Filed: 2026-10-17T16:48:22Z (from `date -u`). Nothing but the tar listing (file names/sizes) and ddad.json (14 KB split file, public metadata) has been opened. No image, point cloud, GT value, depth statistic or model prediction has been read. Changing dataset, bins, metrics or frame list after any output is seen = a new registration.

## Status
DEV, exposed from its first score, report-only: no arm is chosen on it, it never becomes a held-out test without an Auditor ruling; any paper use is labelled "exposed exploratory DEV, N=5 arms disclosed". Scored for all 5 arms (A0, A0b, TS2, TS2+H80, TS2+H300) with the same pinned bench_joint commit on the same host type.

## Dataset and split
- DDAD (TRI), validation split. ddad.json `scene_splits`: "0" = 150 train scenes, "1" = 50 val scenes 000150..000199 (3950 samples). Val = split "1" only. (Argoverse 2 not needed: DDAD obtained.)
- Camera CAMERA_01 (front, 1936x1216) + aggregated Luminar-H2 LIDAR (250 m range) projected into the camera by the official dgp convention (units: metres). Evaluation cap 200 m (packnet-sfm eval_ddad.yaml, camera_01, max depth 200).
- Licence: CC BY-NC-SA 4.0 (non-commercial use only; not redistributed).

## Source and integrity
- URL: https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar (275,266,242,560 B per HEAD; downloaded parts total 275,266,254,848 B incl. tar padding; no sha256 published).
- Published md5 c0da97967f76da80f86d6f97d0d98904 (README). VERIFIED on Isambard: md5 over the 129 parts concatenated in byte order = c0da97967f76da80f86d6f97d0d98904 (job 7497679, rc=0, ended 2026-10-17T10:54:18Z). Parts in /projects/b5dh/scratch_ddad_20261017/parts/.
- Files: tar_listing.txt (119,255 entries) sha256 ef1c81e2f63eea27abac5a9077931ecb004756ab0b14b49c5a2bf7325644d603; ddad.json sha256 14c8aa06c4d262bd962a15caf03970bccf41179267cea1afcf9941e7e93f9c00 (fetched by 14,118-byte range at tar offset 19723312128+512). Per-file sha256 of the extracted subset (SHA256SUMS) is written at extraction and committed.

## Frame list (fixed by rule, from file names only)
- All 3950 CAMERA_01 frames of the 50 val scenes are run contiguously through each model as video (scene = clip; first 160 frames per clip where a window limit applies, as in the existing scorer: stated limitation if a scene exceeds it).
- Scored frames: per scene, CAMERA_01 PNG names sorted ascending (timestamp order), indices 0,10,20,...: 395 frames. List: ddad_val_frames.tsv, sha256 b8d9664fb0c28df0fab61ad71383d48660c5444c4538a8f471fd4bbc8f0e5098. The LIDAR datum for each scored frame is the one paired in the official scene_<sha>.json sample (camera and LIDAR timestamps differ); the scene json is read for pairing only. Script: make_frame_list.py.
- Extract: CAMERA_01 PNG for all 3950 frames, LIDAR npz for the 395 scored frames, scene_*.json, calibration/*.json. Estimated 10-12 GB.
- Decode sanity frames (fixed rule): index 0 of the first 5 scenes in split-file order = (000150, 15616458249936530), (000151, 15609732323309162), (000152, 15609722540809162), (000153, 15616462546436512), (000154, 15609730485809162). They stay in the list (rank 0 is also a scored frame); no frame is dropped except for a documented decode failure. Checks: units (metres), valid mask, LIDAR projection, intrinsics/extrinsics only, GT side, no model output, no depth statistics used for choices.

## Depth bins (declared now, from the LIDAR spec and the 80 m training cap, not from data)
GT valid = projected LIDAR depth > 0 and <= 200 m. Bins (m): [0,20), [20,50), [50,80), [80,120), [120,200]. Per bin report the pixel count and the number of frames contributing.

## Metrics (unaligned: no scale/shift alignment)
Per bin, pooled over scored frames: AbsRel = mean(|pred - gt| / gt); pred:gt = median(pred / gt). Also overall AbsRel over [0,200]. Reported per arm next to the screen decision rule, which does NOT change (primary: VKITTI 80-655 m |1 - pred:gt| reduction). No NF margin is applied to this set; no arm is chosen on it.

## Pinned scorer
bench_joint commit and DEV clip-list sha for the five-arm screen are logged together before the first training run (separate entry, same host type as these scores); a DDAD loader with a regression test (decode, units, masks) is added on outdoor-debug and its commit recorded here before scoring.

## Amendment A1 (2026-10-17T17:06:17Z, from `date -u`; before any model prediction; Auditor O11 R1/R2)
- Resolution rule (frozen): the model prediction for a scored frame is resized BILINEARLY to the native 1936x1216 camera resolution and sampled at the projected LIDAR pixels (integer pixel coordinates of the z-buffered projection); GT is never resized. Metric depth in metres, no scale/shift alignment.
- LIDAR pairing: official scene_<sha>.json samples (camera datum CAMERA_01 <-> LIDAR datum of the same sample). 395 of 395 scored frames paired, 0 missing on disk. Pairs file ddad_val_pairs.tsv sha256 962d499612c40bca7d314f8ff16ce1d6c7b03962ae78938f472a101766150245.
- Extract job: sbatch 7503602, sacct Submit 2026-10-17T16:48:50, Start 2026-10-17T16:48:51 (>= registration filing 16:48:22Z), End 16:53:27, COMPLETED. Extracted 8000 files (3950 CAMERA_01 PNG + 3950 LIDAR npz + 50 scene json + 50 calibration json); sha256sum -c on the cluster 8000/8000 OK. SUMS committed as val_subset_SHA256SUMS.gz (sha256 of the uncompressed file: b12fce8b50ba72c4acecddd194a4b9fdce276c59a7978b68ca18d2815bbbfcea). Local copy under data/DEV_LR/ddad_val/ holds the pruned subset (all 3950 PNGs, the 395 paired LIDAR npz, jsons).
- Earlier job 7497679 (md5 + tar listing only, Start 2026-10-17T10:45:28, End 10:54:21) opened no payload files beyond streaming the tar for md5/listing.
- Decode checks (loader regression test) use only the 5 sanity frames listed above, and cover units, valid mask, projection, intrinsics/extrinsics only.
