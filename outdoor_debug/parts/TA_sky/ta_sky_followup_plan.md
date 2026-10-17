# TA sky follow-up: per-env class + test plan (written BEFORE the visual runs)

Class threshold: median stored depth of seg-sky px (20-frame audit sample, non-finite = +inf) >= 9000 m -> far_dome, < 9000 m -> near_dome, no seg-sky px in sample -> no_seg_sky. Counts: {'near_dome': 14, 'far_dome': 43, 'no_seg_sky': 17}. Gap between classes: largest near median 7584 m (GreatMarsh), smallest far median 11512 m. Guard (Auditor O12.3): sky = seg_sky & (saturated | d >= 9000 m); seg_sky with d < 9000 m -> uncertain (far-dome); near-dome seg-sky = sky only if seg_sky_ok (visual).
Tests: seg-sky visual (O12.4: 13 near-dome + AbandonedCable + 8 other-class + MiddleEast + RetroOffice), saturated-non-sky visual (labelled envs with v2 uncertain > 1%), O10a extension for the 15 unlabelled envs (incl. the 3 currently ON). Frame rule: first trajectory, scan q = 0.2,0.5,0.8,0.1,0.3,0.6,0.9,0.4,0.7, informative = tested set >= 500 px, take the first 3 informative; ALL must be open sky; <2 informative -> not ok. Training environments, not held out: Downtown, Hospital, ForestEnv, OldBrickHouseNight.

| env | class | median seg-sky m | label | v2 unc % | seg-sky test | sat-nonsky test | unlabelled sat test |
|---|---|---|---|---|---|---|---|
| AbandonedCable | near_dome | 3550.0 | sky=44 | 0.035 | True | False | False |
| AbandonedFactory | far_dome | 12376.0 | sky=160 | 0.0 | False | False | False |
| AbandonedFactory2 | far_dome | 65504.0 | sky=171 | 0.0 | False | False | False |
| AbandonedSchool | far_dome | 30832.0 | sky=67 | 0.0 | True | False | False |
| AmericanDiner | far_dome | 65504.0 | sky=0,sky0=242 | 0.0 | False | False | False |
| AmusementPark | near_dome | 316.0 | sky=28 | 0.0 | True | False | False |
| AncientTowns | far_dome | 12072.0 | sky=239 | 0.0 | False | False | False |
| Antiquity3D | far_dome | 12384.0 | sky=7 | 8.811 | True | True | False |
| Apocalyptic | far_dome | 12120.0 | cloud=64,sky=208 | 0.0 | False | False | False |
| ArchVizTinyHouseDay | far_dome | 12120.0 | sky=229 | 0.0 | False | False | False |
| ArchVizTinyHouseNight | no_seg_sky | nan | skysphere=36 | 2.447 | False | True | True |
| BrushifyMoon | far_dome | 65504.0 | sky=59 | 0.075 | False | False | False |
| CarWelding | near_dome | 336.0 | sky=188 | 0.0 | True | False | False |
| CastleFortress | far_dome | 12696.0 | sky=7 | 0.0 | False | False | False |
| CoalMine | no_seg_sky | nan | none | 0.0 | False | False | True |
| ConstructionSite | far_dome | 65504.0 | sky=119 | 0.013 | False | False | False |
| CountryHouse | no_seg_sky | nan | sky=92 | 0.0 | True | False | False |
| CyberPunkDowntown | far_dome | 11824.0 | sky=205 | 0.0 | False | False | False |
| Cyberpunk | far_dome | 12128.0 | sky=155 | 0.0 | False | False | False |
| DesertGasStation | far_dome | 46656.0 | sky=214 | 0.171 | True | False | False |
| Downtown (TE) | near_dome | 339.5 | sky=208 | 0.0 | True | False | False |
| EndofTheWorld | far_dome | 12264.0 | skysphere=64 | 0.0 | False | False | False |
| FactoryWeather | no_seg_sky | nan | none | 12.557 | False | False | True |
| Fantasy | far_dome | 12472.0 | sky=8 | 0.0 | False | False | False |
| ForestEnv (TE) | far_dome | 12456.0 | sky=27 | 0.0 | False | False | False |
| Gascola | far_dome | 12424.0 | sky=59 | 0.0 | False | False | False |
| GothicIsland | far_dome | 12352.0 | sky=242 | 0.0 | False | False | False |
| GreatMarsh | near_dome | 7584.0 | sky=23 | 0.0 | True | False | False |
| HQWesternSaloon | far_dome | 12768.0 | sky=244 | 0.0 | False | False | False |
| HongKong | no_seg_sky | nan | none | 0.205 | False | False | True |
| Hospital (TE) | near_dome | 3090.0 | sky=137 | 0.0 | True | False | False |
| House | far_dome | 13088.0 | sky=85 | 0.0 | False | False | False |
| IndustrialHangar | far_dome | 12512.0 | sky=1 | 0.0 | False | False | False |
| JapaneseAlley | no_seg_sky | nan | none | 2.891 | False | False | True |
| JapaneseCity | far_dome | 12224.0 | sky=217 | 0.0 | False | False | False |
| MiddleEast | far_dome | 12136.0 | sky=48 | 6.275 | True | True | False |
| ModUrbanCity | near_dome | 3198.0 | sky=133 | 0.0 | True | False | False |
| ModernCityDowntown | far_dome | 12536.0 | sky=123 | 0.0 | False | False | False |
| ModularNeighborhood | far_dome | 12552.0 | sky=104 | 0.0 | False | False | False |
| ModularNeighborhoodIntExt | far_dome | 12880.0 | sky=155 | 0.0 | False | False | False |
| NordicHarbor | far_dome | 12240.0 | sky=8 | 0.0 | False | False | False |
| Ocean | no_seg_sky | nan | none | 1.038 | False | False | True |
| Office | far_dome | 12304.0 | skysphere=182 | 0.0 | False | False | False |
| OldBrickHouseDay | near_dome | 58.0625 | sky=30 | 0.0 | True | False | False |
| OldBrickHouseNight (TE) | no_seg_sky | nan | none | 1.831 | False | False | True |
| OldIndustrialCity | far_dome | 12384.0 | sky=34 | 0.0 | False | False | False |
| OldScandinavia | far_dome | 12376.0 | sky=160 | 0.0 | False | False | False |
| OldTownFall | no_seg_sky | nan | none | 14.706 | False | False | True |
| OldTownNight | no_seg_sky | nan | none | 0.0 | False | False | True |
| OldTownSummer | no_seg_sky | nan | none | 14.15 | False | False | True |
| OldTownWinter | no_seg_sky | nan | none | 11.789 | False | False | True |
| PolarSciFi | far_dome | 12776.0 | sky=9 | 0.0 | False | False | False |
| Prison | near_dome | 99.625 | sky=109 | 0.0 | True | False | False |
| Restaurant | far_dome | 12184.0 | sky=70 | 0.0 | False | False | False |
| RetroOffice | near_dome | 2.087890625 | sky=182 | 0.0 | True | False | False |
| Rome | far_dome | 11512.0 | sky=109 | 0.0 | False | False | False |
| Ruins | near_dome | 1832.0 | cloud=36,sky=116 | 0.0 | True | False | False |
| SeasideTown | no_seg_sky | nan | none | 15.359 | False | False | True |
| SeasonalForestAutumn | far_dome | 12656.0 | sky=64 | 0.0 | False | False | False |
| SeasonalForestSpring | far_dome | 12544.0 | sky=64 | 0.0 | False | False | False |
| SeasonalForestSummerNight | far_dome | 12304.0 | sky=64 | 0.0 | False | False | False |
| SeasonalForestWinter | far_dome | 12624.0 | sky=64 | 0.0 | False | False | False |
| SeasonalForestWinterNight | no_seg_sky | nan | none | 24.966 | False | False | True |
| Sewerage | no_seg_sky | nan | none | 0.007 | False | False | True |
| ShoreCaves | near_dome | 275.75 | sky=64 | 0.0 | True | False | False |
| Slaughter | far_dome | 12952.0 | sky=70,skysphere=143 | 0.0 | False | False | False |
| SoulCity | no_seg_sky | nan | sky=175 | 12.437 | True | True | False |
| Supermarket | no_seg_sky | nan | none | 0.766 | False | False | True |
| TerrainBlending | far_dome | 33728.0 | skysphere=36 | 0.0 | True | False | False |
| UrbanConstruction | far_dome | 12256.0 | sky=9 | 0.0 | False | False | False |
| VictorianStreet | far_dome | 12536.0 | sky=146 | 0.0 | False | False | False |
| WaterMillDay | near_dome | 2920.0 | sky=29 | 0.0 | True | False | False |
| WaterMillNight | near_dome | 2974.0 | sky=182 | 0.0 | True | False | False |
| WesternDesertTown | far_dome | 12352.0 | sky=29 | 0.0 | False | False | False |
