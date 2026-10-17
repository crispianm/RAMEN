# TS v2: TartanAir sky / valid recipe proposal (DRAFT; released loader c1abf81 unchanged)

Evidence: SKY_AUDIT_TABLE.md / sky_audit.csv (28 local TRAIN envs, 20 frames each, official tartanairpy 1.4.0 seg; decoding parity with the official reader is exact, tests/test_tartanair_official_parity.py, 118 frames, 28 envs). Downtown, Hospital, ForestEnv, OldBrickHouseNight are training environments, not held out (applies to any TartanAir number on them, including outdoor-debug DEV).

## Problem with the released rule
`sky = ~isfinite(d) | d > 1e4`, `mask = isfinite & d>0 & d<max_depth(1e4)`. In 6 of 24 labelled envs (Downtown, AmusementPark, Prison, ShoreCaves, Hospital, GreatMarsh) the sky is a finite dome < 10 km: ~0% is caught, so it is `sky=0, mask=1` (valid metric depth 36 m-10 km). In 13 more, 1.5-4.9% of sky px lie at 9.4-10.0 km and are likewise `sky=0, mask=1`. A synthetic 8000 m dome therefore trains as valid metric depth (M-03) and exactly 10000.0 is dropped from both sets (B2-03).

## TS v2 definition (per frame; seg-sky first, saturation fallback, uncertain excluded from both)
1. `sat = ~isfinite(d) | d >= 1e4` (inclusive; covers inf, 65504 and the 9.4-10 km dome edge consistently with `d < max_depth`).
2. `seg_sky = seg in sky_ids(env)` from the env seg_label_map.json (names sky, skysphere, sky0, cloud). **sky = seg_sky** whatever its depth (8000 m dome gives sky=1).
3. `uncertain = sat & ~seg_sky` for labelled envs (trees-as-sky #46, glass, unmapped classes such as ArchVizTinyHouseNight id 118): excluded from sky AND from mask.
4. Envs without a sky label (CoalMine, HongKong, Ocean, OldBrickHouseNight; ArchVizTinyHouseNight in practice): saturated px are `uncertain` by default; flagged opt-in `fallback_sat_is_sky=True` makes them sky after a per-env visual check.
5. `mask = isfinite(d) & d>0 & d<max_depth & ~sky & ~uncertain`.
Needs seg_lcam_front for every training env (5.8 GB for the 28 local envs; 74-env pull is a separate Manager/Isambard request, no ssh before go) and the per-env sky-id table; the existing sky loss target (disparity 0 on `sky`) and metric mask both consume (sky, mask).

## id-0 guard (AmericanDiner)
Chosen: keep both sky ids {0, 242} but treat seg id 0 as sky only where saturated (`sat`); id 0 with finite d < 1e4 is uncertain (sky=0, mask=0), so an 'unlabelled' id 0 can never become sky/valid by label alone. Support: over ~400 AmericanDiner frames (every ~4th) id 0 = 419 px, 100% saturated (65504), and id 242 = 0 px, so dropping 0 and using 242 only would lose the only observed sky; HongKong/OldBrickHouseNight have no sky ids so their id 0 stays uncertain (default) as before.

## Regression test (draft)
tests/test_ts_v2_sky_draft.py with outdoor_debug/parts/TA_sky/ts_v2_draft.py: M-03 8000 m dome -> sky=1, mask=0; B2-03 depth 10000.0 -> seg-sky: sky=1; non-sky: uncertain, mask=0; 65504/inf/NaN seg-sky -> sky; saturated tree-labelled px -> uncertain; unlabeled env default and flagged fallback; zero/negative depth not valid. 7 tests pass (incl. id-0 saturated -> sky, id-0 finite -> uncertain, HongKong-like env).

## Caveats
Sample of 20 frames/env, class-id semantics from seg_label_map.json (id 0 doubles as "unlabelled" in HongKong/OldBrickHouseNight and as AmericanDiner's `sky`); no visual verification of the dome/foreground boundary yet; effect on training numbers not measured.
