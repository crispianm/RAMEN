# Visuals index (large images live outside git)
Root: /mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/
- evidence_user_pointed/: horse_real_released_vs_ft_vs_ours.jpg (user's figure, shared log scale 4.05-641.81 m), kitti_sky_released_vs_ours_f109.png
- B1_datasets/: <src>_panels.png (RGB|depth|valid|sky|overlay, 8 clips), <src>_temporal_strip_{0,1}.png, vkitti_sky_overlay_contact_sheet_120frames.png, depth_distribution_across_datasets.png, tartanair_far_categories.png, tartanair_greatmarsh.png, stats_*.csv, tartanair_env_survey.csv
- B2_transforms_masks/: per-source before/after panels, 140-frame sky-overlay contact sheet, ta_sky_encoding_probe.jpg
- D_baseline/: sbs_* (4 outdoor + 2 indoor clips: RGB|GT|4 models|sky mask), pred_vs_gt_outdoor.png, kitti0023_f109_{llb3,b0s0}_{linear,log}.png, kitti_sky_perframe_box.png, horse_f25_log.png
- E_gradients/: class_gradient_share.png (full size + run.log)
- T_trajectory/: sky_vs_step.png, far_ratio_vs_step.png, grad_norm_vs_step.png, panels (horse f25, KITTI 0023 f109 clean+llb3; full-size + panels.npz) in scratch T_trajectory/
- F_overfit/: curves_*.png, panel_<tag>.jpg, full-size panels and raw predictions
