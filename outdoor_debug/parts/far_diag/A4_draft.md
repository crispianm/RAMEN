## A4 (DRAFT, NOT in REGISTRATION.md; to be filed BEFORE the job): released Metric-VDA-S DDAD row at its native 518 input

Status: draft for the Manager/Auditor (Auditor O34 L-3). Report-only, exploratory DEV, never enters a decision. DEBUG ONLY, not paper.

**Purpose.** A2 registered short_side 378 for DDAD so the DDAD far bins share an input scale with the KITTI/VKITTI primary. The release's own default is 518 (`run.py --input_size 518`, README). At 378 a 150 m car is only a few px wide, so part of the release-vs-ours far gap on DDAD (release 120-200 m median pred:gt 0.854 vs ours 0.574-0.548 at 378) could be an input-scale effect. One extra row, release at 518, bounds that. It does not change the arms' rows, which stay at 378.

**Scope (L-3).** The input size changes for the RELEASE spec only. `ours:` stays at 378: `resolve_short_side` raises for any `--ckpt` run with a size other than 378 (forced-fail tested). Ours at 518 is NOT proposed.

**Change (smallest).** New option `--release-short-side {378,518}` (default absent = 378, byte-identical behaviour). Allowed only with a registered `--spec`; `model_input` takes a `short_side` argument (default 378). The result `label` and `meta.short_side` name the size. Patch: `A4_diff.patch` (applies cleanly to LR_DEV/ddad_scorer.py at sha 80916c4f, `git apply --check` OK). Test additions: `A4_test_additions.py` (4 tests, 5 parametrised forced-fail cases; with the existing 14 -> 22 pass on CPU with the patched scorer, checked in a scratch copy; nothing applied to the repo).

**Label (exact, asserted):** "released Metric-VDA-S at its default 518 input (not the registered 378), fp16, exploratory DEV, report-only".

**Pins that change.** New `ddad_scorer.py` sha256 (patched file in scratch: 8e71511b1192a1a1...; final value to be re-computed after the Manager applies the patch); `tests/test_ddad_scorer.py` sha; the sbatch (new `DDAD_RELEASE_MVDA_518.sbatch`: command `--spec metric_vda_s --release-short-side 518`, output dir `.../DDAD_RELEASE_MetricVDA_S_518`, DDAD_SCORER_SHA = the new sha, DDAD_COMMIT = the commit with the patch). **Unchanged:** frames sha b8d9664f, pairs sha 962d4996, bench_joint sha 5d3d048c, release weights sha 3c28432b, bins, A1 resolution rule (bilinear to 1936x1216, GT never resized), fp16, 395 frames / 50 scenes, max_len 160. Existing arm/ref DDAD outputs (scorer 80916c4f) stay valid; the `ours:` path is behaviourally unchanged (tests), so no ours row needs re-running; the Auditor may still ask for a bit-identity re-run of the start row at the new sha (~4 min).

**Auditor checks requested.** (1) diff check of A4_diff.patch; (2) forced-fail: ours + 518, release + 448, release + 0, unknown spec + 518 must raise before any model load (tests do this); (3) first-clip shape log must show model input (N,3,518,826) -> depth same HxW (the O26(1) assert is unchanged); (4) meta.short_side == 518 and the label as above; (5) compare release@378 (7505639) vs release@518 with all other meta equal.

**Cost.** One GH200, 15-min limit. The 378 release run took 208 s; 518 has 1.9x the pixels, so ~5-8 min (inferred, not measured). ~0.1-0.15 NHR.

**Reading rule (pre-declared).** Report the five bins' median pred:gt and pooled AbsRel for release@518 beside release@378. Statement allowed: "the release's 120-200 m median pred:gt is X at 518 vs 0.854 at 378". No statement about ours at 518.
