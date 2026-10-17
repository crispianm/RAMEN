# Unlabelled-env visual decisions (TartanAir TRAIN only, no model output looked at)

Frame rule (fixed): FIRST trajectory (sorted, with depth dir), frames at q = 0.2 / 0.5 / 0.8 of its sorted index. Saturated = ~isfinite(d) | d >= 1e4. Decision rule (Auditor O10a): fallback_sat_is_sky=True ONLY if the saturated pixels are open sky in ALL 3 frames; mixed/unclear/no saturated pixels/interior -> uncertain. Operational reading: a frame with no saturated px, or too few to see (< a few hundred, nothing discernible) counts as 'no evidence' and fails the all-3 test; where a lenient reading would flip the result it is stated. Panels: RGB | saturated mask (red) | log10 depth; /mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_visual_unlabelled/. OldBrickHouseNight is a training environment, not held out (also Downtown, Hospital, ForestEnv elsewhere).

Result: fallback ON for 3 envs (FactoryWeather, SeasonalForestWinterNight, ArchVizTinyHouseNight); uncertain for the other 12. Saturation never occurs in CoalMine (and Sewerage: 1 px). OldTownNight: saturation never occurs but sky is visible at finite depth.

| env | traj | frames | n saturated px (q0.2/0.5/0.8) | decision | reason | panel |
|---|---|---|---|---|---|---|
| CoalMine | P000 | 000119/000298/000476 | 0/0/0 | uncertain | underground interior, never any saturated px (also 0 in the 20-frame audit) | CoalMine.png |
| FactoryWeather | P000 | 000329/000823/001317 | 73401/16374/834 | fallback_sat_is_sky=True | saturated = open sky in all 3 (large sky; narrow sky gaps between buildings in frame 3, 834 px) | FactoryWeather.png |
| HongKong | P000 | 000136/000340/000544 | 11009/0/1185 | uncertain | saturated patches are sky gaps between buildings in frames 1 and 3 but frame 2 has no saturated px (street canyon, no evidence); lenient reading would be on | HongKong.png |
| JapaneseAlley | P000 | 000215/000538/000861 | 2479/6515/33 | uncertain | only corner patches (probably sky gaps) in 1-2; 33 px not discernible in frame 3 | JapaneseAlley.png |
| Ocean | P000 | 000260/000650/001040 | 8304/3737/3017 | uncertain | underwater: saturated px are a thin band at the water surface (11-16 km finite dome), not open sky | Ocean.png |
| OldBrickHouseNight (training environment, not held out) | P001 | 000079/000199/000318 | 0/525/0 | uncertain | training environment, not held out; indoor, 525 px of window-slat at frame 2 only | OldBrickHouseNight.png |
| OldTownFall | P000 | 000127/000317/000508 | 46628/92563/57 | uncertain | frames 1-2 clearly open sky (saturated) but frame 3 has 57 px, none discernible (wall facing); lenient reading would be on | OldTownFall.png |
| OldTownNight | P000 | 000158/000395/000632 | 0/0/0 | uncertain | no saturated px although frames 2-3 show open cloudy sky: sky is a FINITE depth (153/218 m max), not caught by saturation either way | OldTownNight.png |
| OldTownSummer | P000 | 000249/000623/000996 | 1/0/14738 | uncertain | frame 3 sky gaps only; frames 1-2 1 px and 0 px | OldTownSummer.png |
| OldTownWinter | P000 | 000233/000583/000932 | 91949/0/6577 | uncertain | frames 1,3 sky; frame 2 is black/no saturated px; lenient reading would be on | OldTownWinter.png |
| SeasideTown | P000 | 000178/000447/000715 | 1/106997/0 | uncertain | frame 2 open sky (finite dome 11.6-15.6 km), frames 1 and 3 have 1 px and 0 px (interior/alley) | SeasideTown.png |
| SeasonalForestWinterNight | P000 | 000623/001558/002493 | 163921/345/191808 | fallback_sat_is_sky=True | frames 1,3: saturated = open night sky above tree silhouettes (40-47%); frame 2 345 px = canopy gap at the top edge; dark RGB (night) | SeasonalForestWinterNight.png |
| Sewerage | P001 | 000326/000816/001305 | 0/1/0 | uncertain | sewer interior, no sky | Sewerage.png |
| Supermarket | P000 | 000230/000577/000923 | 13060/1299/54 | uncertain | interior: saturated px are freezer glass doors / ceiling light slits, not sky | Supermarket.png |
| ArchVizTinyHouseNight | P000 | 000023/000057/000092 | 7614/60558/5633 | fallback_sat_is_sky=True | skysphere id 36 absent from seg; saturated px = open sky seen through the cabin windows in all 3 frames (13-16 km dome, trees visible); O10a judges the pixels, not the camera position (Auditor O12.2: ON confirmed) | ArchVizTinyHouseNight.png |

(Auditor O12: this STRICT pass is kept for the record; the approved extension rule and its decisions for all 15 envs are in unlabelled_decisions_extension.md/csv.)
