"""DRAFT of the TartanAir sky/valid recipe 'TS v2'.  NOT wired into the released loader (c1abf81 is unchanged).

Per frame: depth d [m] (float32; non-finite or 65504 = saturated), seg s (uint8 class ids), the env's sky class ids.
    sat       = ~isfinite(d) | (d >= SAT_M)              # SAT_M = 1e4; catches inf, 65504 and the 9.4-10 km dome edge
    seg_sky   = isin(s, sky_ids)                         # seg FIRST: sky = seg_sky at any depth (8000 m dome -> sky)
    uncertain = sat & ~seg_sky                           # labelled env: trees-as-sky (#46), glass, unmapped classes
    unlabelled env (no sky ids): uncertain as above, or with fallback_sat_is_sky=True (flagged) sat -> sky, uncertain = 0
    mask      = isfinite(d) & (d > 0) & (d < max_depth) & ~sky & ~uncertain
Returns (sky, mask, uncertain) bool arrays.
"""
import numpy as np

SAT_M = 1.0e4


def ts_v2_sky_valid(depth, seg, sky_ids, max_depth=1.0e4, fallback_sat_is_sky=False):
    d = np.asarray(depth, np.float32)
    sky_ids = set(int(i) for i in sky_ids)
    sat = ~np.isfinite(d) | (d >= SAT_M)
    has_label = len(sky_ids) > 0
    seg_sky = np.isin(seg, list(sky_ids - {0})) if has_label else np.zeros(d.shape, bool)
    # id 0 is 'unlabelled' in HongKong/OldBrickHouseNight but AmericanDiner's `sky`: id 0 counts as sky ONLY where saturated;
    # id 0 with finite depth (< SAT_M) is uncertain (sky=0, mask=0)
    zero_is_sky_id = 0 in sky_ids
    seg_sky |= (seg == 0) & sat & zero_is_sky_id
    zero_finite = (seg == 0) & ~sat & zero_is_sky_id
    sky = seg_sky.copy()
    if has_label or not fallback_sat_is_sky:
        uncertain = sat & ~seg_sky
    else:  # unlabeled env, flagged fallback: saturation == sky
        sky |= sat
        uncertain = np.zeros(d.shape, bool)
    uncertain |= zero_finite
    mask = np.isfinite(d) & (d > 0) & (d < max_depth) & ~sky & ~uncertain
    return sky, mask, uncertain


# ---------------------------------------------------------------------------------------------------------------
# TS v3 guard (Auditor O12.3), NEW function; ts_v2_sky_valid above is unchanged.  Still a DRAFT, not wired into the loader.
FAR_M = 9.0e3     # far-dome class threshold (median seg-sky stored depth per env >= 9000 m), see far_near_class_74.csv


def ts_v2_guarded(depth, seg, sky_ids, far_dome, seg_sky_ok=True, sat_nonsky_is_sky=False, max_depth=1.0e4,
                  fallback_sat_is_sky=False):
    """Per-env flags: far_dome (class), seg_sky_ok (visual check: seg-sky is open sky in all informative fixed frames),
    sat_nonsky_is_sky (visual check: saturated px NOT labelled sky are open sky; default False -> uncertain).
      seg_sky = isin(seg, sky_ids) (id 0 only where saturated, as in ts_v2_sky_valid)
      seg_sky_ok False                      -> every seg-sky px is uncertain (never sky)
      far_dome env, seg_sky_ok True         -> sky = seg_sky & (sat | d >= 9000 m); seg_sky with d < 9000 m -> uncertain
      near_dome env, seg_sky_ok True        -> sky = seg_sky at any depth (as v2)
      saturated & ~seg_sky                  -> uncertain, or sky if sat_nonsky_is_sky (or fallback_sat_is_sky in an unlabelled env)
      mask = finite & d>0 & d<max_depth & ~sky & ~uncertain."""
    d = np.asarray(depth, np.float32)
    ids = set(int(i) for i in sky_ids)
    sat = ~np.isfinite(d) | (d >= SAT_M)
    seg_sky = np.isin(seg, list(ids - {0})) if ids else np.zeros(d.shape, bool)
    seg_sky |= (seg == 0) & sat & (0 in ids)
    zero_finite = (seg == 0) & ~sat & (0 in ids)
    far_ok = sat | (np.nan_to_num(d, nan=0.0, posinf=1e9) >= FAR_M)
    if not seg_sky_ok:
        sky = np.zeros(d.shape, bool); unc = seg_sky.copy()
    elif far_dome:
        sky = seg_sky & far_ok; unc = seg_sky & ~far_ok
    else:
        sky = seg_sky.copy(); unc = np.zeros(d.shape, bool)
    nonsky_sat = sat & ~seg_sky
    if sat_nonsky_is_sky or (fallback_sat_is_sky and not ids):
        sky |= nonsky_sat
    else:
        unc |= nonsky_sat
    unc |= zero_finite
    mask = np.isfinite(d) & (d > 0) & (d < max_depth) & ~sky & ~unc
    return sky, mask, unc
