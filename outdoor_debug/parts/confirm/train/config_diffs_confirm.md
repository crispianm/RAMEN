# Confirmatory from-init arms: resolved-config diffs (BASE = c3rep n46 job 7226119 resolved config, sha d286b62c)
## BASE -> C_TS2 [OK] (5 keys)
- data.seed: 46 -> 50
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- loss.log_terms_every: <absent> -> 10
- training.checkpoint_dir: "runs/dn01_c3rep_20261009_c3_n46" -> "runs/confirm_C_TS2"
## C_TS2 -> C_TS2_H300 [OK] (3 keys)
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/confirm_C_TS2" -> "runs/confirm_C_TS2_H300"
## BASE -> C_TS2_H300 [OK] (7 keys)
- data.seed: 46 -> 50
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- loss.log_terms_every: <absent> -> 10
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/dn01_c3rep_20261009_c3_n46" -> "runs/confirm_C_TS2_H300"
declared value [OK] C_TS2: data.seed = 50 (want 50)
declared value [OK] C_TS2: data.sources[tartanair].ts2_sky = true (want true)
declared value [OK] C_TS2: data.sources[tartanair].ts2_seg_root = "/projects/b5dh/data/train/tartanair_seg/" (want "/projects/b5dh/data/train/tartanair_seg/")
declared value [OK] C_TS2: loss.sky_hinge_weight = 0.0 (want 0.0)
declared value [OK] C_TS2: loss.log_terms = true (want true)
declared value [OK] C_TS2: loss.log_terms_every = 10 (want 10)
declared value [OK] C_TS2: training.checkpoint_dir = "runs/confirm_C_TS2" (want "runs/confirm_C_TS2")
declared value [OK] C_TS2_H300: data.seed = 50 (want 50)
declared value [OK] C_TS2_H300: data.sources[tartanair].ts2_sky = true (want true)
declared value [OK] C_TS2_H300: data.sources[tartanair].ts2_seg_root = "/projects/b5dh/data/train/tartanair_seg/" (want "/projects/b5dh/data/train/tartanair_seg/")
declared value [OK] C_TS2_H300: loss.sky_hinge_weight = 0.1 (want 0.1)
declared value [OK] C_TS2_H300: loss.sky_hinge_min_depth = 300.0 (want 300.0)
declared value [OK] C_TS2_H300: loss.log_terms = true (want true)
declared value [OK] C_TS2_H300: loss.log_terms_every = 10 (want 10)
declared value [OK] C_TS2_H300: training.checkpoint_dir = "runs/confirm_C_TS2_H300" (want "runs/confirm_C_TS2_H300")
fixed recipe value [OK] C_TS2: training.resume_from = null (want null)
fixed recipe value [OK] C_TS2: training.max_steps = 12000 (want 12000)
fixed recipe value [OK] C_TS2: training.lr_decay_steps = 12000 (want 12000)
fixed recipe value [OK] C_TS2: training.warmup_steps = 500 (want 500)
fixed recipe value [OK] C_TS2: training.learning_rate = 0.0001 (want 0.0001)
fixed recipe value [OK] C_TS2: training.checkpoint_at_steps = [6000, 8000, 12000] (want [6000, 8000, 12000])
fixed recipe value [OK] C_TS2: training.seed_scheme = "x1000" (want "x1000")
fixed recipe value [OK] C_TS2: data.split_seed = 43 (want 43)
fixed recipe value [OK] C_TS2: model.vda_init = "released" (want "released")
fixed recipe value [OK] C_TS2: model.init_from = null (want null)
fixed recipe value [OK] C_TS2: model.vda_checkpoint_path = "/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth" (want "/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth")
fixed recipe value [OK] C_TS2_H300: training.resume_from = null (want null)
fixed recipe value [OK] C_TS2_H300: training.max_steps = 12000 (want 12000)
fixed recipe value [OK] C_TS2_H300: training.lr_decay_steps = 12000 (want 12000)
fixed recipe value [OK] C_TS2_H300: training.warmup_steps = 500 (want 500)
fixed recipe value [OK] C_TS2_H300: training.learning_rate = 0.0001 (want 0.0001)
fixed recipe value [OK] C_TS2_H300: training.checkpoint_at_steps = [6000, 8000, 12000] (want [6000, 8000, 12000])
fixed recipe value [OK] C_TS2_H300: training.seed_scheme = "x1000" (want "x1000")
fixed recipe value [OK] C_TS2_H300: data.split_seed = 43 (want 43)
fixed recipe value [OK] C_TS2_H300: model.vda_init = "released" (want "released")
fixed recipe value [OK] C_TS2_H300: model.init_from = null (want null)
fixed recipe value [OK] C_TS2_H300: model.vda_checkpoint_path = "/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth" (want "/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth")

ALL EXPECTED DIFFS/VALUES EXACT: True
