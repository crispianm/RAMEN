"""Writes ta_sky_followup_decisions.csv/.md (all 74 envs), unlabelled_decisions_extension.csv/.md from the frame csvs + visual decisions (made by looking at every panel)."""
import csv, collections
plan = {r["env"]: r for r in csv.DictReader(open("ta_sky_followup_plan.csv"))}
def frames(fn):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(fn)): d[r["env"]].append(r)
    return d
seg, satn, sat = frames("ta_followup_seg_frames.csv"), frames("ta_followup_satnon_frames.csv"), frames("ta_followup_sat_frames.csv")
def used(rs): return "/".join(f"q{r['q']}:{r['frame'][:6]}:{r['n_tested']}" for r in rs if r["used"] == "True")
def n_inf(rs): return sum(r["used"] == "True" for r in rs)
SEG = {  # env: (ok, reason)
 "AbandonedCable": (False, "frames 1-2 open sky but frame 3 (q0.8) seg-sky = frosted window panes with foliage behind glass (finite depth), not open sky"),
 "AbandonedSchool": (True, "3 informative frames: sky through classroom windows (2) and open sky corner (1)"),
 "AmusementPark": (True, "sky in all 3 (thin fog-haze strips below trees read as hazy sky)"),
 "Antiquity3D": (True, "sky in all 3 (frame 3 = 2090 px sky corner); its finite <655 m seg-sky px (12.4% in the audit) are demoted by the far-dome guard, not seen in the fixed frames"),
 "CarWelding": (False, "0 informative frames in 9 (<=2 px of seg-sky; indoor welding hall)"),
 "CountryHouse": (False, "0 seg-sky px in 9 frames (interior); nothing to keep"),
 "DesertGasStation": (True, "open sky in all 3"),
 "Downtown": (True, "sky gaps between buildings/trees in all 3 (training environment, not held out)"),
 "GreatMarsh": (True, "foggy sky in all 3 (fog-white sky above horizon/trees)"),
 "Hospital": (False, "only 1 informative frame in 9 (a 1813 px patch beyond a corridor window): <2 informative (training environment, not held out)"),
 "MiddleEast": (True, "3 informative frames, all sky gaps above the alley"),
 "ModUrbanCity": (True, "hazy sky above the station/rails in all 3 informative frames (0.2, 0.1, 0.3)"),
 "OldBrickHouseDay": (False, "2 informative frames (525/589 px); frame 2 patch is a bright doorway/window interior detail, not discernible as open sky; frame 1 patch not visible; indoor"),
 "Prison": (True, "courtyard sky, sky through barred windows, sky above the wall: all sky"),
 "RetroOffice": (False, "seg-sky = floor/carpet in all 3 frames (harmful label)"),
 "Ruins": (True, "open/cloudy sky in all 3"),
 "ShoreCaves": (True, "overcast sky above cliffs in all 3"),
 "SoulCity": (False, "labelled sky=175 but 0 seg-sky px in 9 frames: seg-sky is not used (sky comes from the saturated test below)"),
 "TerrainBlending": (True, "clear sky in all 3 (dome 24-48 km)"),
 "WaterMillDay": (True, "sky gaps above trees/valley in all 3"),
 "WaterMillNight": (True, "night sky in all 3"),
}
SAT = {  # env: (flag, reason)
 "Antiquity3D": (False, "0 informative frames: every saturated px in the fixed first-trajectory frames is seg-sky; the 'sign'-labelled saturated px (8.8%) are not in these frames -> default uncertain"),
 "MiddleEast": (False, "0 informative frames (<=5 non-sky saturated px in 9 frames; id-0 saturated px not in fixed frames) -> default uncertain"),
 "SoulCity": (True, "3 informative frames (139042/40520/4983 px), all open sky between buildings (frame 3 = grey sky sliver, zoomed)"),
 "ArchVizTinyHouseNight": (True, "3 informative frames, open sky through the cabin windows (O12.2: pixels judged, not the camera position)"),
}
rows = []
for e, p in sorted(plan.items()):
    c = p["cls"]; unc = float(p["v2_uncertain_pct"])
    ok = SEG.get(e, (None, ""))
    s_ok = ""; reason = ""
    if e in SEG: s_ok, reason = ok
    elif c == "far_dome": s_ok, reason = True, "not visually tested (far-dome; the guard sky = seg_sky & (sat | d >= 9000 m) applies)"
    else: s_ok, reason = "n/a", "no seg-sky label px"
    sn = SAT.get(e)
    rows.append(dict(env=e, status=p["status"], cls=c, median_seg_sky_m=p["median_seg_sky_m"], sky_label=p["sky_label"],
        seg_sky_tested=e in SEG, seg_frames_used=used(seg.get(e, [])) if e in SEG else "", seg_sky_ok=s_ok, seg_reason=reason,
        v2_uncertain_pct=unc, satnonsky_tested=e in SAT, satnonsky_frames_used=used(satn.get(e, [])) if e in SAT else "",
        sat_nonsky_is_sky=(sn[0] if sn else ("" if unc == 0 else "False (default uncertain; share <= 1% or unlabelled-env path)")), satnonsky_reason=sn[1] if sn else ""))
with open("ta_sky_followup_decisions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cnt = collections.Counter((r["cls"], str(r["seg_sky_ok"])) for r in rows)
with open("ta_sky_followup_decisions.md", "w") as f:
    f.write("# TA sky follow-up: per-env class + visual decisions (all 74)\n\nGuard (Auditor O12.3): sky = seg_sky & (saturated | d >= 9000 m); seg_sky with d < 9000 m -> uncertain (far-dome); near-dome seg-sky = sky only if seg_sky_ok. seg_sky_ok False -> all seg-sky px uncertain (never sky). Frame rule: first trajectory, q-scan 0.2,0.5,0.8,0.1,0.3,0.6,0.9,0.4,0.7, informative >= 500 tested px, first 3 informative, ALL must be sky, <2 informative -> not ok. Panels: /mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_visual_neardome/ and TA_visual_satnonsky/. Training environments, not held out: Downtown, Hospital, ForestEnv, OldBrickHouseNight. Frames used: q:frame:tested px.\n\n")
    f.write("| env | class | seg_sky_ok | seg frames (q:frame:px) | reason | sat-non-sky is sky | sat reason |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        if r["seg_sky_tested"] or r["satnonsky_tested"]:
            f.write(f"| {r['env']}{' (TE)' if 'training' in r['status'] else ''} | {r['cls']} | {r['seg_sky_ok']} | {r['seg_frames_used']} | {r['seg_reason']} | {r['sat_nonsky_is_sky'] if r['satnonsky_tested'] else ''} | {r['satnonsky_reason']} |\n")
    f.write("\nAll other envs: far-dome (not visually tested): seg_sky_ok=True with the guard; no_seg_sky unlabelled envs: see unlabelled_decisions_extension.md; saturated non-sky share <= 1%: default uncertain. Full per-env rows in ta_sky_followup_decisions.csv.\n")
# ---- unlabelled extension
EXT = {
 "ArchVizTinyHouseNight": ("fallback_sat_is_sky=True", "sky through the cabin windows in all 3 (O12.2: ON confirmed; pixels judged)"),
 "CoalMine": ("uncertain", "0 saturated px in 9 frames (underground)"),
 "FactoryWeather": ("fallback_sat_is_sky=True", "open sky above/between buildings in all 3 (frame 3 = 834 px sky sliver at top edge)"),
 "HongKong": ("uncertain", "3 informative frames (q0.2, 0.8, 0.6) but the saturated patches are black/very dark night regions between buildings, not discernible as open sky (borderline; extension frame q0.6 is a dark alley)"),
 "JapaneseAlley": ("fallback_sat_is_sky=True", "3 informative frames (q0.2, 0.5, 0.3): pink/purple dusk haze sky above the lanterns/roofs (corner patches; zoomed)"),
 "Ocean": ("uncertain", "underwater: saturated px = thin band at the water surface, not open sky"),
 "OldBrickHouseNight": ("uncertain", "training environment, not held out; only 2 informative frames (q0.5, 0.7; 525/589 px), interior window slats/doorway, not sky"),
 "OldTownFall": ("fallback_sat_is_sky=True", "3 informative frames (q0.2, 0.5, 0.1): open sky in all (q0.8 had 57 px -> replaced by q0.1)"),
 "OldTownNight": ("uncertain", "0 saturated px in 9 frames (sky at finite depth; see oldtownnight_farplane.md)"),
 "OldTownSummer": ("fallback_sat_is_sky=True", "3 informative frames (q0.8, 0.1, 0.9): open sky in all (q0.2/0.5 had 1/0 px)"),
 "OldTownWinter": ("fallback_sat_is_sky=True", "3 informative frames (q0.2, 0.8, 0.1): open sky in all (q0.5 had 0 px)"),
 "SeasideTown": ("fallback_sat_is_sky=True", "3 informative frames (q0.5, 0.1, 0.6): hazy sky in all (q0.2/0.8 had 1/0 px)"),
 "SeasonalForestWinterNight": ("fallback_sat_is_sky=True", "3 informative frames (q0.2, 0.8, 0.1): night sky above ridge/trees in all (q0.5 had 345 px -> replaced)"),
 "Sewerage": ("uncertain", "<= 9 saturated px in any of 9 frames (sewer interior)"),
 "Supermarket": ("uncertain", "interior: saturated px are freezer glass doors / ceiling light slits in all 3 informative frames (q0.2, 0.5, 0.3)"),
}
STRICT = {r["env"]: r for r in csv.DictReader(open("unlabelled_decisions.csv"))}
out = []
for e, (dec, why) in EXT.items():
    rs = sat[e]
    out.append(dict(env=e, status="training environment, not held out" if e == "OldBrickHouseNight" else "cluster TRAIN env", strict_decision=STRICT[e]["decision"],
        extension_decision=dec, changed_vs_strict=STRICT[e]["decision"] != dec, frames_scanned_q_npx="/".join(f"{r['q']}:{r['n_tested']}" for r in rs),
        frames_used_q_frame_npx=used(rs), n_informative=n_inf(rs), reason=why, panel=f"TA_visual_unlabelled_ext/{e}.png"))
with open("unlabelled_decisions_extension.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
with open("unlabelled_decisions_extension.md", "w") as f:
    f.write("# Unlabelled-env decisions, extension rule (Auditor O12.1: APPROVED, applied to all 15 incl. the 3 previously ON)\n\nBoth passes are kept: STRICT (unlabelled_decisions.md/csv, fixed q=0.2/0.5/0.8, any <500 px frame fails the all-3 test) and EXTENSION (this file). Reason for the change: the strict reading failed envs only because a fixed frame was uninformative (no saturated px visible), which says nothing about whether saturated px are sky; the extension replaces uninformative frames (<500 saturated px) in the order q=0.1,0.3,0.6,0.9,0.4,0.7 of the first trajectory until 3 informative frames or the list runs out; ALL informative frames must be open sky; mixed -> uncertain; <2 informative -> uncertain. Saturated = ~isfinite(d) | d >= 1e4. Panels (RGB | seg-sky overlay | saturated red | log10 depth, informative frames only): /mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_visual_unlabelled_ext/. OldBrickHouseNight is a training environment, not held out.\n\n")
    f.write("| env | strict | extension | changed | q:sat px scanned | frames used (q:frame:px) | reason |\n|---|---|---|---|---|---|---|\n")
    for r in out: f.write(f"| {r['env']} | {r['strict_decision']} | {r['extension_decision']} | {r['changed_vs_strict']} | {r['frames_scanned_q_npx']} | {r['frames_used_q_frame_npx']} | {r['reason']} |\n")
    ch = [r["env"] for r in out if r["changed_vs_strict"]]
    f.write(f"\nChanged vs strict: {', '.join(ch)}. Unchanged ON: ArchVizTinyHouseNight, FactoryWeather, SeasonalForestWinterNight. HongKong stays uncertain.\n")
print([r["env"] for r in out if r["changed_vs_strict"]], cnt)
