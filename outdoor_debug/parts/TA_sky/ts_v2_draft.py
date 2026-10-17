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
