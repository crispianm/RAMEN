"""Pre-run per-env class + test-plan table (Auditor O12.4). Reads far_near_class_74.csv, sky_audit_74.csv. Writes ta_sky_followup_plan.csv/.md BEFORE any visual run."""
import csv
cls = {r["env"]: r for r in csv.DictReader(open("far_near_class_74.csv"))}
aud = {r["env"]: r for r in csv.DictReader(open("sky_audit_74.csv"))}
OTHER8 = {"AbandonedCable", "AbandonedSchool", "DesertGasStation", "TerrainBlending", "Antiquity3D", "ArchVizTinyHouseNight", "CountryHouse", "SoulCity"}
UNL15 = {"CoalMine", "FactoryWeather", "HongKong", "JapaneseAlley", "Ocean", "OldBrickHouseNight", "OldTownFall", "OldTownNight", "OldTownSummer", "OldTownWinter", "SeasideTown", "SeasonalForestWinterNight", "Sewerage", "Supermarket", "ArchVizTinyHouseNight"}
TE = {"Downtown", "Hospital", "ForestEnv", "OldBrickHouseNight"}
rows = []
for e in sorted(cls):
    c = cls[e]["cls"]; unc = float(aud[e]["v2_uncertain_pct_of_px"]); lab = aud[e]["sky_label"]
    near = c == "near_dome"
    seg = near or e in OTHER8 or e == "MiddleEast"
    if e in UNL15: seg = False  # no usable seg-sky label; handled by the saturated test
    satnon = (lab != "none") and unc > 1.0
    sat = e in UNL15
    rows.append(dict(env=e, status="training environment, not held out" if e in TE else "cluster TRAIN env", cls=c, median_seg_sky_m=cls[e]["median_m"],
        sky_label=lab, v2_uncertain_pct=unc, test_seg_sky=seg, test_satnonsky=satnon, test_sat_unlabelled_O10a_ext=sat,
        pre_run_default=("seg_sky_ok pending" if seg else ("far-dome guard applies (seg_sky_ok=True)" if c == "far_dome" else "n/a")),
        decision_seg_sky_ok="", decision_sat_nonsky="uncertain (default)" if unc > 0 else "n/a", decision_fallback_sat_is_sky=""))
with open("ta_sky_followup_plan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
n = Counter(r["cls"] for r in rows)
with open("ta_sky_followup_plan.md", "w") as f:
    f.write("# TA sky follow-up: per-env class + test plan (written BEFORE the visual runs)\n\nClass threshold: median stored depth of seg-sky px (20-frame audit sample, non-finite = +inf) >= 9000 m -> far_dome, < 9000 m -> near_dome, no seg-sky px in sample -> no_seg_sky. "
            f"Counts: {dict(n)}. Gap between classes: largest near median 7584 m (GreatMarsh), smallest far median 11512 m. Guard (Auditor O12.3): sky = seg_sky & (saturated | d >= 9000 m); seg_sky with d < 9000 m -> uncertain (far-dome); near-dome seg-sky = sky only if seg_sky_ok (visual).\n"
            "Tests: seg-sky visual (O12.4: 13 near-dome + AbandonedCable + 8 other-class + MiddleEast + RetroOffice), saturated-non-sky visual (labelled envs with v2 uncertain > 1%), O10a extension for the 15 unlabelled envs (incl. the 3 currently ON). Frame rule: first trajectory, scan q = 0.2,0.5,0.8,0.1,0.3,0.6,0.9,0.4,0.7, informative = tested set >= 500 px, take the first 3 informative; ALL must be open sky; <2 informative -> not ok. Training environments, not held out: Downtown, Hospital, ForestEnv, OldBrickHouseNight.\n\n")
    f.write("| env | class | median seg-sky m | label | v2 unc % | seg-sky test | sat-nonsky test | unlabelled sat test |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows: f.write(f"| {r['env']}{' (TE)' if r['env'] in TE else ''} | {r['cls']} | {r['median_seg_sky_m']} | {r['sky_label']} | {r['v2_uncertain_pct']} | {r['test_seg_sky']} | {r['test_satnonsky']} | {r['test_sat_unlabelled_O10a_ext']} |\n")
print(n, sum(r["test_seg_sky"] for r in rows), sum(r["test_satnonsky"] for r in rows), sum(r["test_sat_unlabelled_O10a_ext"] for r in rows))
