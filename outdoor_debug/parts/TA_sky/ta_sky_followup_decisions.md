# TA sky follow-up: per-env class + visual decisions (all 74)

Guard (Auditor O12.3): sky = seg_sky & (saturated | d >= 9000 m); seg_sky with d < 9000 m -> uncertain (far-dome); near-dome seg-sky = sky only if seg_sky_ok. seg_sky_ok False -> all seg-sky px uncertain (never sky). Frame rule: first trajectory, q-scan 0.2,0.5,0.8,0.1,0.3,0.6,0.9,0.4,0.7, informative >= 500 tested px, first 3 informative, ALL must be sky, <2 informative -> not ok. Panels: /mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_visual_neardome/ and TA_visual_satnonsky/. Training environments, not held out: Downtown, Hospital, ForestEnv, OldBrickHouseNight. Frames used: q:frame:tested px.

| env | class | seg_sky_ok | seg frames (q:frame:px) | reason | sat-non-sky is sky | sat reason |
|---|---|---|---|---|---|---|
| AbandonedCable | near_dome | False | q0.2:000544:23565/q0.5:001361:138250/q0.8:002177:12675 | frames 1-2 open sky but frame 3 (q0.8) seg-sky = frosted window panes with foliage behind glass (finite depth), not open sky |  |  |
| AbandonedSchool | far_dome | True | q0.2:000271:904/q0.1:000135:1310/q0.9:001223:17755 | 3 informative frames: sky through classroom windows (2) and open sky corner (1) |  |  |
| AmusementPark | near_dome | True | q0.2:000310:72325/q0.5:000776:152471/q0.8:001242:82444 | sky in all 3 (thin fog-haze strips below trees read as hazy sky) |  |  |
| Antiquity3D | far_dome | True | q0.2:000298:225561/q0.5:000745:146924/q0.8:001192:2090 | sky in all 3 (frame 3 = 2090 px sky corner); its finite <655 m seg-sky px (12.4% in the audit) are demoted by the far-dome guard, not seen in the fixed frames | False | 0 informative frames: every saturated px in the fixed first-trajectory frames is seg-sky; the 'sign'-labelled saturated px (8.8%) are not in these frames -> default uncertain |
| ArchVizTinyHouseNight | no_seg_sky | n/a |  | no seg-sky label px | True | 3 informative frames, open sky through the cabin windows (O12.2: pixels judged, not the camera position) |
| CarWelding | near_dome | False |  | 0 informative frames in 9 (<=2 px of seg-sky; indoor welding hall) |  |  |
| CountryHouse | no_seg_sky | False |  | 0 seg-sky px in 9 frames (interior); nothing to keep |  |  |
| DesertGasStation | far_dome | True | q0.2:000124:99523/q0.5:000311:16566/q0.8:000497:46355 | open sky in all 3 |  |  |
| Downtown (TE) | near_dome | True | q0.2:000559:4914/q0.5:001399:1383/q0.8:002238:31985 | sky gaps between buildings/trees in all 3 (training environment, not held out) |  |  |
| GreatMarsh | near_dome | True | q0.2:000707:138103/q0.5:001768:140544/q0.8:002828:243348 | foggy sky in all 3 (fog-white sky above horizon/trees) |  |  |
| Hospital (TE) | near_dome | False | q0.5:001117:1813 | only 1 informative frame in 9 (a 1813 px patch beyond a corridor window): <2 informative (training environment, not held out) |  |  |
| MiddleEast | far_dome | True | q0.8:000951:4679/q0.1:000118:18129/q0.3:000356:1142 | 3 informative frames, all sky gaps above the alley | False | 0 informative frames (<=5 non-sky saturated px in 9 frames; id-0 saturated px not in fixed frames) -> default uncertain |
| ModUrbanCity | near_dome | True | q0.2:000419:103833/q0.1:000209:49998/q0.3:000629:14738 | hazy sky above the station/rails in all 3 informative frames (0.2, 0.1, 0.3) |  |  |
| OldBrickHouseDay | near_dome | False | q0.5:000199:525/q0.7:000278:589 | 2 informative frames (525/589 px); frame 2 patch is a bright doorway/window interior detail, not discernible as open sky; frame 1 patch not visible; indoor |  |  |
| Prison | near_dome | True | q0.2:000181:45084/q0.5:000453:9077/q0.8:000724:57782 | courtyard sky, sky through barred windows, sky above the wall: all sky |  |  |
| RetroOffice | near_dome | False | q0.2:000025:62215/q0.5:000064:79213/q0.8:000102:30459 | seg-sky = floor/carpet in all 3 frames (harmful label) |  |  |
| Ruins | near_dome | True | q0.2:000806:111572/q0.5:002016:140146/q0.8:003226:211373 | open/cloudy sky in all 3 |  |  |
| ShoreCaves | near_dome | True | q0.2:000343:112063/q0.5:000858:46285/q0.8:001373:101223 | overcast sky above cliffs in all 3 |  |  |
| SoulCity | no_seg_sky | False |  | labelled sky=175 but 0 seg-sky px in 9 frames: seg-sky is not used (sky comes from the saturated test below) | True | 3 informative frames (139042/40520/4983 px), all open sky between buildings (frame 3 = grey sky sliver, zoomed) |
| TerrainBlending | far_dome | True | q0.2:000494:138020/q0.8:001978:127076/q0.1:000247:178094 | clear sky in all 3 (dome 24-48 km) |  |  |
| WaterMillDay | near_dome | True | q0.2:000267:9810/q0.8:001070:62310/q0.1:000133:20136 | sky gaps above trees/valley in all 3 |  |  |
| WaterMillNight | near_dome | True | q0.2:000228:66052/q0.5:000571:182933/q0.8:000913:12153 | night sky in all 3 |  |  |

All other envs: far-dome (not visually tested): seg_sky_ok=True with the guard; no_seg_sky unlabelled envs: see unlabelled_decisions_extension.md; saturated non-sky share <= 1%: default uncertain. Full per-env rows in ta_sky_followup_decisions.csv.
