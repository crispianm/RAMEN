# v3 change manifest (modified pre-existing paths vs ef1268b)

every MODIFIED (not added) pre-existing path from the v3 base to the package, with the reason; tests/test_dn01_v3_change_manifest.py fails on a modified / deleted / renamed path that is not listed here (it replaces the whole-tree guard the FIX2 re-pin no longer provides). Added files are new work and not listed.

| path | reason |
|---|---|
| `bench_joint.py` | WP22-24 adapters `dav2_s_ft:` / `daac_ft:` / `da3_s_ft:` (PERFRAME_FT, _perframe_ft) + `metric_vda_s_ft:` (FT_VDA, _vda_ft) in split_spec / build_pipeline; the GPU agf timing branch in depth_post_stages; (headc port) the _ours depth_input branch. Scoring maths unchanged (noisestress AST pin). + the `agf_backend` key in the run / latency artifacts. |
| `data/irs_dataset.py` | WP11: `val_fraction <= 0` is a TRUE zero hold-out (every scene trains; a val split then raises); > 0 and the default keep the old at-least-one-scene rule. Tested (tests/test_dn01_v3_syn_data.py). |
| `data/video_clip_dataset.py` | WP20 llvar-train: opt-in data.lowlight keys kappa_mult_choices / blur_px / profile (_llvar_opts); absent = byte-identical. Tested (tests/test_dn01_v3_llvar_train.py). |
| `data/vda_semantics_dataset.py` | `_VDADegradedMixin.sparse_gt` class attr + per-sample `sparse` flag (WP13); default None emits nothing (byte-identical). |
| `docs/DN01_JOB_LEDGER.md` | Ledger text appended by the headc port cherry-pick (design-(c) package entry); documentation only. |
| `evaluation/depth_probe.py` | WP24: `depth_agf_torch` + `AGF_BACKEND` (env RAMEN_AGF_BACKEND, default cpu = the unchanged numpy reference); make_post routes agf tokens by backend. |
| `models/__init__.py` | build_model branches `dav2` (WP22), `daac` (WP23), `da3` (WP22b); headc port: depth_input / stage2_width kwargs on SharedTemporalNet beside the METRICFIX unfreeze_head. |
| `models/shared_temporal.py` | headc port (design (c)): depth_input / stage2_width, gates, restore_batch / consume (default-off; step 0 bit-identical, tested). |
| `models/vda_net.py` | WP21: VDANet.depth_head_parameters() so head_lr_mult works for a plain arch=vda (FT Metric-VDA-S). |
| `scripts/dn01_bvi_fnoise.py` | WP24: one key `agf_backend` in the result artifact; scoring unchanged. |
| `scripts/dn01_heldout_eval.py` | WP24: one key `agf_backend` in the result artifact (cpu = numpy reference | gpu); scoring unchanged. |
| `tests/test_dn01_e4_keys.py` | config-prefix allowlist gains dn01_v3_ / dn01_headc_ (the repo-wide key-allowlist pins, as fx2 / mf did). |
| `tests/test_dn01_fix2.py` | re-pinned to the audited FIX2 package commit (32c709b..c3fd200) instead of the working tree; reason in runs/WP2/fix2_repin.md. |
| `tests/test_dn01_fix2_score.py` | re-pinned to the FIX2 score commit (c3fd200..0560c6a) instead of HEAD; reason in runs/WP2/fix2_repin.md. |
| `tests/test_dn01_metricfix.py` | the METRICFIX no-new-keys pin skips dn01_v3_ configs (they inherit the a1h loss / eval block). |
| `tests/test_dn01_noisestress.py` | bench_joint AST pin whitelist: the per-frame / FT adapters (PERFRAME_FT, _perframe_ft, split_spec, build_pipeline, FT_VDA, _vda_ft, _ours); spec grammar only. |
| `tests/test_dn01_rc.py` | train.py / models / data code-delta pin: new files and tokens for protocol v3 WP12-14, 21-23, 22b and the headc keys. |
| `tests/test_dn01_sky_configs.py` | sky-keys config allowlist gains dn01_v3_ / dn01_headc_ prefixes. |
| `train.py` | WP13: source type kitti_raw / nyu_train / scannetpp_train / spring (V3_REAL_SOURCES), sparse_gt flag + check_sparse_gt + sparse_ssi_loss + per-sample routing in _compute_loss_vda; WP21: head_lr_mult guard accepts plain arch=vda; WP22-23: VDA_LOSS_ARCHES; headc port: gate_lr_scale group (+ the WP17 fix when head_lr_mult is set). All default-off / additive. |

**Stray files:** Before: the only check that saw uncommitted / stray files was the untracked-file loop of tests/test_dn01_fix2.py (dropped by the re-pin); the rc code-delta pin and the noisestress pins read `git diff` of TRACKED paths, so they never saw untracked files. Now: tests/test_dn01_v3_change_manifest.py::test_a_clone_has_no_untracked_stray_files fails on any untracked non-ignored file.
