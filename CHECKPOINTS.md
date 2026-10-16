# Checkpoints

Checkpoints will be released with the paper under the same terms as the code; the download URLs below are placeholders. `configs/paper_results.yaml` (section `models:`) pins every checkpoint by sha256 and refuses a missing or mismatching file. The 16-hex prefixes below are copied from that config; the full sha256 is in the config.

| Model | Role | Training config | sha256 (first 16) | Download |
|---|---|---|---|---|
| RAMEN, seed 45, step 12000 | headline TEST checkpoint (continuation lineage, see README) | `dn01_v3_synreal_r50_B` then `dn01_v3_bfix_C3_B` | `c9decd90a4b4c274` | TODO at release |
| RAMEN, seed 46, step 12000 | multi-seed | `dn01_c3rep_20261009_c3_n46` | `f5c7c8f98fe0c9f9` | TODO at release |
| RAMEN, seed 47, step 12000 | multi-seed | `dn01_c3rep_20261009_c3_n47` | `2ee98b7ee25bd6d5` | TODO at release |
| RAMEN, seed 48, step 12000 | multi-seed | `dn01_c3rep_20261009_c3_n48` | `bf053bd38fdb97d5` | TODO at release |
| Metric-VDA-S fine-tune, seed 46 | matched depth-only baseline | `dn01_c3rep_20261009_ft_n46` | `18c9d8caeab1ab88` | TODO at release |
| Metric-VDA-S fine-tune, seed 47 | matched depth-only baseline | `dn01_c3rep_20261009_ft_n47` | `ec38c3179cb63349` | TODO at release |
| Metric-VDA-S fine-tune, seed 48 | matched depth-only baseline | `dn01_c3rep_20261009_ft_n48` | `d769e513c6e8995c` | TODO at release |
| DAv2-S fine-tune, seed 46 | per-frame baseline | `dn01_c3rep_pf_dav2a_n46` | `e5cbf7ca420741aa` | TODO at release |
| DA-AC fine-tune (metric head), seed 46 | per-frame baseline | `dn01_c3rep_pf_daacm_n46` | `665b59ff0ad82486` | TODO at release |
| Restorer stage initialisation (`st_temp2_init`) | RAMEN training initialisation | `scripts/dn01_st_distill.py` | `96d35f08ecd5c988` | TODO at release |

Third-party weights (not redistributed here), with the upstream source named in `configs/paper_results.yaml`:
Metric-Video-Depth-Anything-Small `3c28432b4e1f0d7b`, Metric-Video-Depth-Anything-Large `24eba25342e7ee0f`, Depth-Anything-V2-Small `715fade13be8f229`, Depth-Anything-AC (ViT-S) `d5c802e195c59e68`.
