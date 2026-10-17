# Stage 1 resolved-config diffs (flat keys). BASE per pair = that start's own c3rep resolved config; yaml in configs/arm_<run>.yaml
## S1_n46_TS2 -> S1_n46_H300 [OK] (3 keys)
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/confirm1_S1_n46_TS2" -> "runs/confirm1_S1_n46_H300"
## S1_n47_TS2 -> S1_n47_H300 [OK] (3 keys)
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/confirm1_S1_n47_TS2" -> "runs/confirm1_S1_n47_H300"
## S1_n48_TS2 -> S1_n48_H300 [OK] (3 keys)
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1
- training.checkpoint_dir: "runs/confirm1_S1_n48_TS2" -> "runs/confirm1_S1_n48_H300"
## S1_n46_TS2 -> S1_n47_TS2 [OK] (4 keys)
- data.seed: 50 -> 51
- training.checkpoint_dir: "runs/confirm1_S1_n46_TS2" -> "runs/confirm1_S1_n47_TS2"
- training.resume_from: "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt" -> "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt"
- training.resume_reseed: 50 -> 51
## S1_n46_TS2 -> S1_n48_TS2 [OK] (4 keys)
- data.seed: 50 -> 52
- training.checkpoint_dir: "runs/confirm1_S1_n46_TS2" -> "runs/confirm1_S1_n48_TS2"
- training.resume_from: "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt" -> "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt"
- training.resume_reseed: 50 -> 52
## S1_n47_TS2 -> S1_n48_TS2 [OK] (4 keys)
- data.seed: 51 -> 52
- training.checkpoint_dir: "runs/confirm1_S1_n47_TS2" -> "runs/confirm1_S1_n48_TS2"
- training.resume_from: "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt" -> "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt"
- training.resume_reseed: 51 -> 52
## S1_n46_H300 -> S1_n47_H300 [OK] (4 keys)
- data.seed: 50 -> 51
- training.checkpoint_dir: "runs/confirm1_S1_n46_H300" -> "runs/confirm1_S1_n47_H300"
- training.resume_from: "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt" -> "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt"
- training.resume_reseed: 50 -> 51
## S1_n46_H300 -> S1_n48_H300 [OK] (4 keys)
- data.seed: 50 -> 52
- training.checkpoint_dir: "runs/confirm1_S1_n46_H300" -> "runs/confirm1_S1_n48_H300"
- training.resume_from: "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt" -> "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt"
- training.resume_reseed: 50 -> 52
## S1_n47_H300 -> S1_n48_H300 [OK] (4 keys)
- data.seed: 51 -> 52
- training.checkpoint_dir: "runs/confirm1_S1_n47_H300" -> "runs/confirm1_S1_n48_H300"
- training.resume_from: "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt" -> "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt"
- training.resume_reseed: 51 -> 52
declared value [OK] S1_n46_TS2: data.seed = 50 (want 50)
declared value [OK] S1_n46_TS2: training.resume_reseed = 50 (want 50)
declared value [OK] S1_n46_TS2: training.resume_from = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt" (want "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt")
declared value [OK] S1_n46_TS2: training.max_steps = 14000 (want 14000)
declared value [OK] S1_n46_TS2: training.lr_decay_steps = 14000 (want 14000)
declared value [OK] S1_n46_TS2: training.lr_restart_step = 12000 (want 12000)
declared value [OK] S1_n46_TS2: training.lr_restart_warmup = 100 (want 100)
declared value [OK] S1_n46_TS2: training.lr_restart_peak = 1.0 (want 1.0)
declared value [OK] S1_n46_TS2: training.learning_rate = 0.0001 (want 0.0001)
declared value [OK] S1_n46_TS2: training.warmup_steps = 500 (want 500)
declared value [OK] S1_n46_TS2: training.checkpoint_at_steps = [12500, 13000, 13500, 14000] (want [12500, 13000, 13500, 14000])
declared value [OK] S1_n46_TS2: training.checkpoint_dir = "runs/confirm1_S1_n46_TS2" (want "runs/confirm1_S1_n46_TS2")
declared value [OK] S1_n46_TS2: loss.log_terms = true (want true)
declared value [OK] S1_n46_TS2: loss.log_terms_every = 10 (want 10)
declared value [OK] S1_n46_TS2: data.sources[tartanair].ts2_sky = true (want true)
declared value [OK] S1_n46_TS2: data.sources[tartanair].ts2_seg_root = "/projects/b5dh/data/train/tartanair_seg/" (want "/projects/b5dh/data/train/tartanair_seg/")
declared value [OK] S1_n46_TS2: data.sources[vkitti].vkitti_max_depth = 655 (want 655)
declared value [OK] S1_n46_TS2: data.split_seed = 43 (want 43)
declared value [OK] S1_n46_TS2: training.seed_scheme = "x1000" (want "x1000")
declared value [OK] S1_n46_TS2: loss.sky_hinge_weight = 0.0 (want 0.0)
declared value [OK] S1_n46_H300: data.seed = 50 (want 50)
declared value [OK] S1_n46_H300: training.resume_reseed = 50 (want 50)
declared value [OK] S1_n46_H300: training.resume_from = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt" (want "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt")
declared value [OK] S1_n46_H300: training.max_steps = 14000 (want 14000)
declared value [OK] S1_n46_H300: training.lr_decay_steps = 14000 (want 14000)
declared value [OK] S1_n46_H300: training.lr_restart_step = 12000 (want 12000)
declared value [OK] S1_n46_H300: training.lr_restart_warmup = 100 (want 100)
declared value [OK] S1_n46_H300: training.lr_restart_peak = 1.0 (want 1.0)
declared value [OK] S1_n46_H300: training.learning_rate = 0.0001 (want 0.0001)
declared value [OK] S1_n46_H300: training.warmup_steps = 500 (want 500)
declared value [OK] S1_n46_H300: training.checkpoint_at_steps = [12500, 13000, 13500, 14000] (want [12500, 13000, 13500, 14000])
declared value [OK] S1_n46_H300: training.checkpoint_dir = "runs/confirm1_S1_n46_H300" (want "runs/confirm1_S1_n46_H300")
declared value [OK] S1_n46_H300: loss.log_terms = true (want true)
declared value [OK] S1_n46_H300: loss.log_terms_every = 10 (want 10)
declared value [OK] S1_n46_H300: data.sources[tartanair].ts2_sky = true (want true)
declared value [OK] S1_n46_H300: data.sources[tartanair].ts2_seg_root = "/projects/b5dh/data/train/tartanair_seg/" (want "/projects/b5dh/data/train/tartanair_seg/")
declared value [OK] S1_n46_H300: data.sources[vkitti].vkitti_max_depth = 655 (want 655)
declared value [OK] S1_n46_H300: data.split_seed = 43 (want 43)
declared value [OK] S1_n46_H300: training.seed_scheme = "x1000" (want "x1000")
declared value [OK] S1_n46_H300: loss.sky_hinge_weight = 0.1 (want 0.1)
declared value [OK] S1_n46_H300: loss.sky_hinge_min_depth = 300.0 (want 300.0)
declared value [OK] S1_n47_TS2: data.seed = 51 (want 51)
declared value [OK] S1_n47_TS2: training.resume_reseed = 51 (want 51)
declared value [OK] S1_n47_TS2: training.resume_from = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt" (want "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt")
declared value [OK] S1_n47_TS2: training.max_steps = 14000 (want 14000)
declared value [OK] S1_n47_TS2: training.lr_decay_steps = 14000 (want 14000)
declared value [OK] S1_n47_TS2: training.lr_restart_step = 12000 (want 12000)
declared value [OK] S1_n47_TS2: training.lr_restart_warmup = 100 (want 100)
declared value [OK] S1_n47_TS2: training.lr_restart_peak = 1.0 (want 1.0)
declared value [OK] S1_n47_TS2: training.learning_rate = 0.0001 (want 0.0001)
declared value [OK] S1_n47_TS2: training.warmup_steps = 500 (want 500)
declared value [OK] S1_n47_TS2: training.checkpoint_at_steps = [12500, 13000, 13500, 14000] (want [12500, 13000, 13500, 14000])
declared value [OK] S1_n47_TS2: training.checkpoint_dir = "runs/confirm1_S1_n47_TS2" (want "runs/confirm1_S1_n47_TS2")
declared value [OK] S1_n47_TS2: loss.log_terms = true (want true)
declared value [OK] S1_n47_TS2: loss.log_terms_every = 10 (want 10)
declared value [OK] S1_n47_TS2: data.sources[tartanair].ts2_sky = true (want true)
declared value [OK] S1_n47_TS2: data.sources[tartanair].ts2_seg_root = "/projects/b5dh/data/train/tartanair_seg/" (want "/projects/b5dh/data/train/tartanair_seg/")
declared value [OK] S1_n47_TS2: data.sources[vkitti].vkitti_max_depth = 655 (want 655)
declared value [OK] S1_n47_TS2: data.split_seed = 43 (want 43)
declared value [OK] S1_n47_TS2: training.seed_scheme = "x1000" (want "x1000")
declared value [OK] S1_n47_TS2: loss.sky_hinge_weight = 0.0 (want 0.0)
declared value [OK] S1_n47_H300: data.seed = 51 (want 51)
declared value [OK] S1_n47_H300: training.resume_reseed = 51 (want 51)
declared value [OK] S1_n47_H300: training.resume_from = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt" (want "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n47/step12000.pt")
declared value [OK] S1_n47_H300: training.max_steps = 14000 (want 14000)
declared value [OK] S1_n47_H300: training.lr_decay_steps = 14000 (want 14000)
declared value [OK] S1_n47_H300: training.lr_restart_step = 12000 (want 12000)
declared value [OK] S1_n47_H300: training.lr_restart_warmup = 100 (want 100)
declared value [OK] S1_n47_H300: training.lr_restart_peak = 1.0 (want 1.0)
declared value [OK] S1_n47_H300: training.learning_rate = 0.0001 (want 0.0001)
declared value [OK] S1_n47_H300: training.warmup_steps = 500 (want 500)
declared value [OK] S1_n47_H300: training.checkpoint_at_steps = [12500, 13000, 13500, 14000] (want [12500, 13000, 13500, 14000])
declared value [OK] S1_n47_H300: training.checkpoint_dir = "runs/confirm1_S1_n47_H300" (want "runs/confirm1_S1_n47_H300")
declared value [OK] S1_n47_H300: loss.log_terms = true (want true)
declared value [OK] S1_n47_H300: loss.log_terms_every = 10 (want 10)
declared value [OK] S1_n47_H300: data.sources[tartanair].ts2_sky = true (want true)
declared value [OK] S1_n47_H300: data.sources[tartanair].ts2_seg_root = "/projects/b5dh/data/train/tartanair_seg/" (want "/projects/b5dh/data/train/tartanair_seg/")
declared value [OK] S1_n47_H300: data.sources[vkitti].vkitti_max_depth = 655 (want 655)
declared value [OK] S1_n47_H300: data.split_seed = 43 (want 43)
declared value [OK] S1_n47_H300: training.seed_scheme = "x1000" (want "x1000")
declared value [OK] S1_n47_H300: loss.sky_hinge_weight = 0.1 (want 0.1)
declared value [OK] S1_n47_H300: loss.sky_hinge_min_depth = 300.0 (want 300.0)
declared value [OK] S1_n48_TS2: data.seed = 52 (want 52)
declared value [OK] S1_n48_TS2: training.resume_reseed = 52 (want 52)
declared value [OK] S1_n48_TS2: training.resume_from = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt" (want "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt")
declared value [OK] S1_n48_TS2: training.max_steps = 14000 (want 14000)
declared value [OK] S1_n48_TS2: training.lr_decay_steps = 14000 (want 14000)
declared value [OK] S1_n48_TS2: training.lr_restart_step = 12000 (want 12000)
declared value [OK] S1_n48_TS2: training.lr_restart_warmup = 100 (want 100)
declared value [OK] S1_n48_TS2: training.lr_restart_peak = 1.0 (want 1.0)
declared value [OK] S1_n48_TS2: training.learning_rate = 0.0001 (want 0.0001)
declared value [OK] S1_n48_TS2: training.warmup_steps = 500 (want 500)
declared value [OK] S1_n48_TS2: training.checkpoint_at_steps = [12500, 13000, 13500, 14000] (want [12500, 13000, 13500, 14000])
declared value [OK] S1_n48_TS2: training.checkpoint_dir = "runs/confirm1_S1_n48_TS2" (want "runs/confirm1_S1_n48_TS2")
declared value [OK] S1_n48_TS2: loss.log_terms = true (want true)
declared value [OK] S1_n48_TS2: loss.log_terms_every = 10 (want 10)
declared value [OK] S1_n48_TS2: data.sources[tartanair].ts2_sky = true (want true)
declared value [OK] S1_n48_TS2: data.sources[tartanair].ts2_seg_root = "/projects/b5dh/data/train/tartanair_seg/" (want "/projects/b5dh/data/train/tartanair_seg/")
declared value [OK] S1_n48_TS2: data.sources[vkitti].vkitti_max_depth = 655 (want 655)
declared value [OK] S1_n48_TS2: data.split_seed = 43 (want 43)
declared value [OK] S1_n48_TS2: training.seed_scheme = "x1000" (want "x1000")
declared value [OK] S1_n48_TS2: loss.sky_hinge_weight = 0.0 (want 0.0)
declared value [OK] S1_n48_H300: data.seed = 52 (want 52)
declared value [OK] S1_n48_H300: training.resume_reseed = 52 (want 52)
declared value [OK] S1_n48_H300: training.resume_from = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt" (want "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n48/step12000.pt")
declared value [OK] S1_n48_H300: training.max_steps = 14000 (want 14000)
declared value [OK] S1_n48_H300: training.lr_decay_steps = 14000 (want 14000)
declared value [OK] S1_n48_H300: training.lr_restart_step = 12000 (want 12000)
declared value [OK] S1_n48_H300: training.lr_restart_warmup = 100 (want 100)
declared value [OK] S1_n48_H300: training.lr_restart_peak = 1.0 (want 1.0)
declared value [OK] S1_n48_H300: training.learning_rate = 0.0001 (want 0.0001)
declared value [OK] S1_n48_H300: training.warmup_steps = 500 (want 500)
declared value [OK] S1_n48_H300: training.checkpoint_at_steps = [12500, 13000, 13500, 14000] (want [12500, 13000, 13500, 14000])
declared value [OK] S1_n48_H300: training.checkpoint_dir = "runs/confirm1_S1_n48_H300" (want "runs/confirm1_S1_n48_H300")
declared value [OK] S1_n48_H300: loss.log_terms = true (want true)
declared value [OK] S1_n48_H300: loss.log_terms_every = 10 (want 10)
declared value [OK] S1_n48_H300: data.sources[tartanair].ts2_sky = true (want true)
declared value [OK] S1_n48_H300: data.sources[tartanair].ts2_seg_root = "/projects/b5dh/data/train/tartanair_seg/" (want "/projects/b5dh/data/train/tartanair_seg/")
declared value [OK] S1_n48_H300: data.sources[vkitti].vkitti_max_depth = 655 (want 655)
declared value [OK] S1_n48_H300: data.split_seed = 43 (want 43)
declared value [OK] S1_n48_H300: training.seed_scheme = "x1000" (want "x1000")
declared value [OK] S1_n48_H300: loss.sky_hinge_weight = 0.1 (want 0.1)
declared value [OK] S1_n48_H300: loss.sky_hinge_min_depth = 300.0 (want 300.0)
## screen arm_TS2 -> S1_n46_TS2 [OK]: ['data.seed', 'training.checkpoint_dir', 'training.resume_reseed']
## screen arm_TS2_H300 -> S1_n46_H300 [OK]: ['data.seed', 'training.checkpoint_dir', 'training.resume_reseed']
## screen arm_TS2 -> S1_n47_TS2 [OK]: ['data.seed', 'training.checkpoint_dir', 'training.resume_from', 'training.resume_reseed']
## screen arm_TS2_H300 -> S1_n47_H300 [OK]: ['data.seed', 'training.checkpoint_dir', 'training.resume_from', 'training.resume_reseed']
## screen arm_TS2 -> S1_n48_TS2 [OK]: ['data.seed', 'training.checkpoint_dir', 'training.resume_from', 'training.resume_reseed']
## screen arm_TS2_H300 -> S1_n48_H300 [OK]: ['data.seed', 'training.checkpoint_dir', 'training.resume_from', 'training.resume_reseed']

ALL EXPECTED DIFFS/VALUES EXACT: True
