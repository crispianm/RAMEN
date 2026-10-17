# Resolved-config diffs (flat keys; yaml written to configs/arm_<name>.yaml)

A0 vs effective_c3_seed45.yaml (base): differs ONLY by the common continuation fields (resume_from, max_steps, checkpoint_at_steps, checkpoint_dir, LR keys, log_terms_every): True

- loss.log_terms_every: <absent> -> 10
- training.checkpoint_at_steps: [100, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000] -> [12500, 13000, 13500, 14000]
- training.checkpoint_dir: "runs/dn01_v3_bfix_C3_B" -> "runs/outdoor_screen_A0"
- training.lr_decay_steps: 12000 -> 14000
- training.lr_restart_peak: <absent> -> 1.0
- training.lr_restart_step: <absent> -> 12000
- training.lr_restart_warmup: <absent> -> 100
- training.max_steps: 12000 -> 14000
- training.resume_from: "/projects/b5dh/RAMEN_train_dn01_v3_synreal_r50_B/runs/dn01_v3_synreal_r50_B/step6000.pt" -> "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt"
LR-key identical across 6 arms [OK] training.learning_rate = {'0.0001'}
LR-key identical across 6 arms [OK] training.lr_decay_steps = {'14000'}
LR-key identical across 6 arms [OK] training.lr_restart_peak = {'1.0'}
LR-key identical across 6 arms [OK] training.lr_restart_step = {'12000'}
LR-key identical across 6 arms [OK] training.lr_restart_warmup = {'100'}
LR-key identical across 6 arms [OK] training.max_steps = {'14000'}
LR-key identical across 6 arms [OK] training.warmup_steps = {'500'}
log_terms/log_terms_every per arm: {'A0': (True, 10), 'A0b': (True, 10), 'TS2': (True, 10), 'TS2_H80': (True, 10), 'TS2_H300': (True, 10), 'TS2_VK655': (True, 10)}


## A0 -> A0b  [OK]  (3 differing keys)
- data.seed: 45 -> 49
- training.checkpoint_dir: "runs/outdoor_screen_A0" -> "runs/outdoor_screen_A0b"
- training.resume_reseed: <absent> -> 49

## A0 -> TS2  [OK]  (3 differing keys)
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- training.checkpoint_dir: "runs/outdoor_screen_A0" -> "runs/outdoor_screen_TS2"

## TS2 -> TS2_H80  [OK]  (3 differing keys)
- loss.sky_hinge_min_depth: <absent> -> 80.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/outdoor_screen_TS2" -> "runs/outdoor_screen_TS2_H80"

## TS2 -> TS2_H300  [OK]  (3 differing keys)
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/outdoor_screen_TS2" -> "runs/outdoor_screen_TS2_H300"

## TS2 -> TS2_VK655  [OK]  (3 differing keys)
- data.sources[vkitti].vkitti_max_depth: 655 -> 656
- data.sources[vkitti].vkitti_sky_as_valid_depth: <absent> -> true
- training.checkpoint_dir: "runs/outdoor_screen_TS2" -> "runs/outdoor_screen_TS2_VK655"

## A0 -> TS2_H80  [info]  (5 differing keys)
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- loss.sky_hinge_min_depth: <absent> -> 80.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/outdoor_screen_A0" -> "runs/outdoor_screen_TS2_H80"

## A0 -> TS2_H300  [info]  (5 differing keys)
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/outdoor_screen_A0" -> "runs/outdoor_screen_TS2_H300"

## A0 -> TS2_VK655  [info]  (5 differing keys)
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- data.sources[vkitti].vkitti_max_depth: 655 -> 656
- data.sources[vkitti].vkitti_sky_as_valid_depth: <absent> -> true
- training.checkpoint_dir: "runs/outdoor_screen_A0" -> "runs/outdoor_screen_TS2_VK655"
declared value [OK] A0: data.seed = '45' (want 45)
declared value [OK] A0b: data.seed = '49' (want 49)
declared value [OK] A0b: training.resume_reseed = '49' (want 49)
declared value [OK] TS2: data.sources[tartanair].ts2_sky = 'true' (want True)
declared value [OK] TS2_H80: data.sources[tartanair].ts2_sky = 'true' (want True)
declared value [OK] TS2_H80: loss.sky_hinge_weight = '0.1' (want 0.1)
declared value [OK] TS2_H80: loss.sky_hinge_min_depth = '80.0' (want 80.0)
declared value [OK] TS2_H300: data.sources[tartanair].ts2_sky = 'true' (want True)
declared value [OK] TS2_H300: loss.sky_hinge_weight = '0.1' (want 0.1)
declared value [OK] TS2_H300: loss.sky_hinge_min_depth = '300.0' (want 300.0)
declared value [OK] TS2_VK655: data.sources[tartanair].ts2_sky = 'true' (want True)
declared value [OK] TS2_VK655: data.sources[vkitti].vkitti_sky_as_valid_depth = 'true' (want True)
declared value [OK] TS2_VK655: data.sources[vkitti].vkitti_max_depth = '656' (want 656)

ALL EXPECTED DIFFS EXACT: True
