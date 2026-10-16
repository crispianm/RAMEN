| key | C3 s45 | RAMEN n46 | FT n46 |
|---|---|---|---|
| data.seed | 45 | 46 | 46 |
| loss.enh_weight | 1.0 | 1.0 | 0.0 |
| model.shared_temporal | True | True | <absent> |
| model.st_depth_input | restored | restored | <absent> |
| model.st_inject | False | False | <absent> |
| model.st_stage2_width | wide2x | wide2x | <absent> |
| model.st_temp2_init | /projects/b5dh/RAMEN_train_dn01_st_recal/runs/recal_full/temp2_distilled.pt | /projects/b5dh/RAMEN_train_dn01_st_recal/runs/recal_full/temp2_distilled.pt | <absent> |
| model.st_unfreeze_motion | True | True | <absent> |
| model.st_width | full | full | <absent> |
| training.checkpoint_at_steps | [100, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000] | [6000, 8000, 12000] | [6000, 8000, 12000] |
| training.checkpoint_dir | runs/dn01_v3_bfix_C3_B | runs/dn01_c3rep_20261009_c3_n46 | runs/dn01_c3rep_20261009_ft_n46 |
| training.gate_lr_scale | 10 | 10 | <absent> |
| training.resume_from | /projects/b5dh/RAMEN_train_dn01_v3_synreal_r50_B/runs/dn01_v3_synreal_r50_B/step6000.pt | None | None |
