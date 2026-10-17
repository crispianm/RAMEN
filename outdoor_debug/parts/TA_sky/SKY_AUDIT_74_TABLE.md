# TartanAir 74-env sky audit (cluster copy; 20 frames/env, rule identical to sky_audit.py)

Data: cluster TRAIN tree /projects/b5dh/data/train/tartanair (depth, read only) + official tartanairpy 1.4.0 seg (sha-verified); label maps = tartanairpy 158a684 segfiles.zip sha256 80a87e58...9468 (matches Auditor). Full columns in sky_audit_74.csv; sky ids in sky_ids_74.json. TE = training environment, not held out. `rule` = released loader sky rule. v2 columns = TS v2 draft (id-0 guard, no fallback): uncertain % of px; any-change % of px = px whose (sky, mask) differs from released or that is uncertain. 28 local-env rows are identical in every column to sky_audit.csv.

| env | TE | class | sky label | sky %px | sky depth min/mode/max (m) | rule catches % of seg-sky | seg-sky finite <655 m % | seg-sky 655-1e4 % | non-sky rule px %px (top classes) | v2 uncertain %px | v2 any-change %px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AbandonedCable |  | OTHER | sky=44 | 11.645 | 2310.0 / 2916.0 / 59904.0 | 36.432 | 0.0 | 63.568 | 0.035 id0:100% | 0.035 | 7.438 |
| AbandonedFactory |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=160 | 20.385 | 9440.0 / 11616.0 / 16400.0 | 97.877 | 0.0 | 2.123 | 0.0  | 0.0 | 0.433 |
| AbandonedFactory2 |  | 65504 | sky=171 | 7.263 | 2.708984375 / 65504.0 / 65504.0 | 99.906 | 0.094 | 0.0 | 0.0  | 0.0 | 0.007 |
| AbandonedSchool |  | OTHER | sky=67 | 1.131 | 13.2265625 / 34368.0 / 40032.0 | 99.775 | 0.225 | 0.0 | 0.0 landscape:100% | 0.0 | 0.003 |
| AmericanDiner |  | 65504 | sky=0,sky0=242 | 0.001 | 65504.0 / 65504.0 / 65504.0 | 100.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 |
| AmusementPark |  | near dome <10 km | sky=28 | 27.559 | 221.625 / 300.5 / 434.0 | 0.0 | 100.0 | 0.0 | 0.0  | 0.0 | 27.559 |
| AncientTowns |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=239 | 10.843 | 9440.0 / 11616.0 / 16360.0 | 97.539 | 0.0 | 2.461 | 0.0  | 0.0 | 0.267 |
| Antiquity3D |  | OTHER | sky=7 | 23.587 | 0.84228515625 / 11608.0 / 16384.0 | 86.41 | 12.376 | 1.214 | 8.806 sign:100% | 8.811 | 12.016 |
| Apocalyptic |  | ~11.6 km dome (1.5-4.9% under 10 km) | cloud=64,sky=208 | 3.951 | 8.1015625 / 11720.0 / 16320.0 | 96.257 | 0.238 | 3.506 | 0.0  | 0.0 | 0.148 |
| ArchVizTinyHouseDay |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=229 | 2.509 | 8.9453125 / 11632.0 / 16344.0 | 97.38 | 0.07 | 2.55 | 0.0  | 0.0 | 0.066 |
| ArchVizTinyHouseNight |  | label present, 0 sky px in sample | skysphere=36 | 0.0 | nan / nan / nan |  |  |  | 2.446 id118:100% | 2.447 | 2.447 |
| BrushifyMoon |  | 65504 | sky=59 | 42.522 | 65504.0 / 65504.0 / 65504.0 | 100.0 | 0.0 | 0.0 | 0.075 earth:87%; sun:13% | 0.075 | 0.075 |
| CarWelding |  | near dome <10 km | sky=188 | 0.0 | 257.25 / 257.25 / 395.0 | 0.0 | 100.0 | 0.0 | 0.0  | 0.0 | 0.0 |
| CastleFortress |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=7 | 17.697 | 239.75 / 11656.0 / 16384.0 | 98.315 | 0.0 | 1.685 | 0.0  | 0.0 | 0.298 |
| CoalMine |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 0.0  | 0.0 | 0.0 |
| ConstructionSite |  | 65504 | sky=119 | 14.231 | 65504.0 / 65504.0 / 65504.0 | 100.0 | 0.0 | 0.0 | 0.013 ground:100% | 0.013 | 0.013 |
| CountryHouse |  | label present, 0 sky px in sample | sky=92 | 0.0 | nan / nan / nan |  |  |  | 0.0  | 0.0 | 0.0 |
| CyberPunkDowntown |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=205 | 0.558 | 9448.0 / 11560.0 / 16328.0 | 96.275 | 0.0 | 3.725 | 0.0  | 0.0 | 0.021 |
| Cyberpunk |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=155 | 5.581 | 9464.0 / 11632.0 / 16384.0 | 98.002 | 0.0 | 1.998 | 0.0  | 0.0 | 0.111 |
| DesertGasStation |  | OTHER | sky=214 | 25.618 | 34560.0 / 42560.0 / 60096.0 | 100.0 | 0.0 | 0.0 | 0.171 ground:100% | 0.171 | 0.171 |
| Downtown | TE | near dome <10 km | sky=208 | 5.547 | 233.0 / 331.75 / 510.5 | 0.0 | 100.0 | 0.0 | 0.0  | 0.0 | 5.547 |
| EndofTheWorld |  | ~11.6 km dome (1.5-4.9% under 10 km) | skysphere=64 | 17.801 | 9416.0 / 11584.0 / 16344.0 | 97.882 | 0.0 | 2.118 | 0.0  | 0.0 | 0.377 |
| FactoryWeather |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 12.557 id0:100% | 12.557 | 12.557 |
| Fantasy |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=8 | 28.308 | 9424.0 / 11608.0 / 16384.0 | 98.147 | 0.0 | 1.853 | 0.0  | 0.0 | 0.525 |
| ForestEnv | TE | ~11.6 km dome (1.5-4.9% under 10 km) | sky=27 | 20.728 | 10.0859375 / 11616.0 / 16448.0 | 97.949 | 0.001 | 2.05 | 0.0 ground:100% | 0.0 | 0.425 |
| Gascola |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=59 | 11.679 | 9264.0 / 11680.0 / 16384.0 | 97.506 | 0.0 | 2.494 | 0.0  | 0.0 | 0.291 |
| GothicIsland |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=242 | 10.81 | 9408.0 / 11632.0 / 16416.0 | 98.255 | 0.0 | 1.745 | 0.0  | 0.0 | 0.189 |
| GreatMarsh |  | near dome <10 km | sky=23 | 29.833 | 2.404296875 / 8208.0 / 10152.0 | 0.149 | 0.351 | 99.5 | 0.0  | 0.0 | 29.789 |
| HQWesternSaloon |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=244 | 19.196 | 9456.0 / 11576.0 / 16384.0 | 98.379 | 0.0 | 1.621 | 0.0  | 0.0 | 0.311 |
| HongKong |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 0.205 id0:100% | 0.205 | 0.205 |
| Hospital | TE | near dome <10 km | sky=137 | 0.01 | 30.96875 / 3210.0 / 3570.0 | 0.0 | 0.122 | 99.878 | 0.0  | 0.0 | 0.01 |
| House |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=85 | 0.553 | 9928.0 / 13112.0 / 15704.0 | 99.943 | 0.0 | 0.057 | 0.0  | 0.0 | 0.0 |
| IndustrialHangar |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=1 | 23.67 | 9400.0 / 11592.0 / 16448.0 | 98.128 | 0.0 | 1.872 | 0.0  | 0.0 | 0.443 |
| JapaneseAlley |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 2.891 id0:100% | 2.891 | 2.891 |
| JapaneseCity |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=217 | 3.187 | 9448.0 / 11568.0 / 16376.0 | 96.631 | 0.0 | 3.369 | 0.0  | 0.0 | 0.107 |
| MiddleEast |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=48 | 3.886 | 9448.0 / 11640.0 / 16152.0 | 97.408 | 0.0 | 2.592 | 6.275 id0:100% | 6.275 | 6.376 |
| ModUrbanCity |  | near dome <10 km | sky=133 | 6.626 | 2.244140625 / 2902.0 / 4116.0 | 0.0 | 1.208 | 98.792 | 0.0  | 0.0 | 6.626 |
| ModernCityDowntown |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=123 | 6.964 | 9448.0 / 11584.0 / 16400.0 | 98.295 | 0.0 | 1.705 | 0.0  | 0.0 | 0.119 |
| ModularNeighborhood |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=104 | 20.354 | 105.4375 / 11616.0 / 16448.0 | 98.228 | 0.0 | 1.771 | 0.0 landscape:100% | 0.0 | 0.361 |
| ModularNeighborhoodIntExt |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=155 | 16.658 | 4.7890625 / 11608.0 / 16416.0 | 98.542 | 0.008 | 1.45 | 0.0  | 0.0 | 0.243 |
| NordicHarbor |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=8 | 12.995 | 9400.0 / 11584.0 / 16328.0 | 97.997 | 0.0 | 2.003 | 0.0  | 0.0 | 0.26 |
| Ocean |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 1.038 z:100%; ground:0% | 1.038 | 1.038 |
| Office |  | ~11.6 km dome (1.5-4.9% under 10 km) | skysphere=182 | 0.486 | 9456.0 / 12064.0 / 15528.0 | 96.395 | 0.0 | 3.605 | 0.0  | 0.0 | 0.018 |
| OldBrickHouseDay |  | near dome <10 km | sky=30 | 7.563 | 36.75 / 66.0 / 82.4375 | 0.0 | 100.0 | 0.0 | 0.0  | 0.0 | 7.563 |
| OldBrickHouseNight | TE | no label | none | 0.0 | nan / nan / nan |  |  |  | 1.831 id0:100% | 1.831 | 1.831 |
| OldIndustrialCity |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=34 | 18.823 | 9368.0 / 11688.0 / 16400.0 | 97.735 | 0.0 | 2.265 | 0.0  | 0.0 | 0.426 |
| OldScandinavia |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=160 | 13.321 | 9448.0 / 11624.0 / 16464.0 | 97.851 | 0.0 | 2.149 | 0.0  | 0.0 | 0.286 |
| OldTownFall |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 14.706 z:100% | 14.706 | 14.706 |
| OldTownNight |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 0.0  | 0.0 | 0.0 |
| OldTownSummer |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 14.15 light:100% | 14.15 | 14.15 |
| OldTownWinter |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 11.789 light:100% | 11.789 | 11.789 |
| PolarSciFi |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=9 | 26.105 | 9456.0 / 11608.0 / 16416.0 | 98.47 | 0.0 | 1.53 | 0.0  | 0.0 | 0.399 |
| Prison |  | near dome <10 km | sky=109 | 23.611 | 35.53125 / 86.6875 / 176.625 | 0.0 | 100.0 | 0.0 | 0.0  | 0.0 | 23.611 |
| Restaurant |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=70 | 1.899 | 9464.0 / 11560.0 / 16352.0 | 97.46 | 0.0 | 2.54 | 0.0  | 0.0 | 0.048 |
| RetroOffice |  | near dome <10 km | sky=182 | 13.421 | 0.68798828125 / 2.115234375 / 206.5 | 0.0 | 100.0 | 0.0 | 0.0  | 0.0 | 13.421 |
| Rome |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=109 | 3.704 | 9448.0 / 11592.0 / 15416.0 | 95.144 | 0.0 | 4.856 | 0.0  | 0.0 | 0.18 |
| Ruins |  | near dome <10 km | cloud=36,sky=116 | 26.273 | 55.65625 / 2050.0 / 2716.0 | 0.0 | 0.113 | 99.887 | 0.0  | 0.0 | 26.273 |
| SeasideTown |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 15.35 bp:100% | 15.359 | 15.359 |
| SeasonalForestAutumn |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=64 | 26.979 | 3.69140625 / 11632.0 / 16464.0 | 98.363 | 0.001 | 1.636 | 0.0  | 0.0 | 0.442 |
| SeasonalForestSpring |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=64 | 27.407 | 33.34375 / 11648.0 / 16512.0 | 98.12 | 0.009 | 1.871 | 0.0  | 0.0 | 0.515 |
| SeasonalForestSummerNight |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=64 | 19.738 | 49.40625 / 11600.0 / 16448.0 | 96.989 | 0.003 | 3.008 | 0.0  | 0.0 | 0.594 |
| SeasonalForestWinter |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=64 | 33.994 | 21.09375 / 11640.0 / 16464.0 | 98.286 | 0.002 | 1.712 | 0.0  | 0.0 | 0.583 |
| SeasonalForestWinterNight |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 24.951 bp:100% | 24.966 | 24.966 |
| Sewerage |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 0.007 id0:100% | 0.007 | 0.007 |
| ShoreCaves |  | near dome <10 km | sky=64 | 34.188 | 90.5625 / 261.75 / 498.0 | 0.0 | 100.0 | 0.0 | 0.0  | 0.0 | 34.188 |
| Slaughter |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=70,skysphere=143 | 9.168 | 9432.0 / 11584.0 / 16376.0 | 98.426 | 0.0 | 1.574 | 0.0  | 0.0 | 0.144 |
| SoulCity |  | label present, 0 sky px in sample | sky=175 | 0.0 | nan / nan / nan |  |  |  | 12.437 id0:100% | 12.437 | 12.437 |
| Supermarket |  | no label | none | 0.0 | nan / nan / nan |  |  |  | 0.766 id0:100% | 0.766 | 0.766 |
| TerrainBlending |  | OTHER | skysphere=36 | 23.086 | 23728.0 / 32800.0 / 48384.0 | 100.0 | 0.0 | 0.0 | 0.0  | 0.0 | 0.0 |
| UrbanConstruction |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=9 | 8.834 | 9440.0 / 11632.0 / 16360.0 | 98.278 | 0.0 | 1.722 | 0.0  | 0.0 | 0.152 |
| VictorianStreet |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=146 | 10.123 | 9448.0 / 11648.0 / 16400.0 | 98.367 | 0.0 | 1.633 | 0.0  | 0.0 | 0.165 |
| WaterMillDay |  | near dome <10 km | sky=29 | 14.605 | 2076.0 / 2896.0 / 4248.0 | 0.0 | 0.0 | 100.0 | 0.0  | 0.0 | 14.605 |
| WaterMillNight |  | near dome <10 km | sky=182 | 21.178 | 2060.0 / 2830.0 / 4228.0 | 0.0 | 0.0 | 100.0 | 0.0  | 0.0 | 21.178 |
| WesternDesertTown |  | ~11.6 km dome (1.5-4.9% under 10 km) | sky=29 | 8.026 | 2.013671875 / 11608.0 / 16384.0 | 98.048 | 0.023 | 1.929 | 0.0  | 0.0 | 0.157 |

## Class counts (74 envs)
~11.6 km dome with 1.5-4.9% under 10 km: 35; near dome <10 km: 13 (AmusementPark, CarWelding*, Downtown, GreatMarsh, Hospital, ModUrbanCity, OldBrickHouseDay, Prison, RetroOffice*, Ruins, ShoreCaves, WaterMillDay, WaterMillNight); 65504: 4 (AbandonedFactory2, AmericanDiner, BrushifyMoon, ConstructionSite); no label: 14; NOT in the four classes: 8 (below). *tiny/odd sky, see flags.

## Envs outside the four behaviour classes (flags)
- AbandonedCable (sky=44, 11.6% px): dome 2.3-2.9 km mode but max 59904 m; rule catches only 36.4%, 63.6% sits at 655-1e4 m (mixed near-dome + far tail); 0.035% saturated px are unlabelled id 0.
- AbandonedSchool (sky=67): dome 34 km (mode 34368, max 40032), rule catches 99.8%: finite far dome >1e4 (not inf/65504).
- DesertGasStation (sky=214): finite 34-60 km dome, caught 100%; 0.17% of px non-sky ground classes saturated.
- TerrainBlending (skysphere=36): finite 23.7-48 km dome, caught 100%.
- Antiquity3D (sky=7, 23.6% px): BIG finite share: 12.4% of seg-sky px finite < 655 m (min 0.84 m, p1 1.2 m), 1.2% 655-1e4, 86.4% caught; plus 8.8% of ALL px are saturated but labelled `sign` (class mislabel; becomes uncertain under TS v2). Label is not clean.
- ArchVizTinyHouseNight (skysphere=36): 0 sky px in the 20 sampled frames; 2.4% px saturated (finite 13.6-16.4 km dome) labelled id 118 (unmapped) -> uncertain; see unlabelled_decisions.md (visually sky through windows).
- CountryHouse (sky=92): label present, 0 sky px in 20 frames, 0 saturation (interior-ish sample).
- SoulCity (sky=175): label present but 0 seg-sky px in 20 frames while 12.4% px saturated, all id 0 (unlabelled) -> uncertain under v2. A labelled env whose sky is unlabelled.
- Also id-0-saturated in labelled envs (not in the sky ids): MiddleEast 6.3% px (v2 uncertain 6.3%, any change 6.4%), AbandonedCable 0.035%.
- Odd near-class: RetroOffice sky=182 (13.4% px) at 0.69-206 m, mode 2.1 m; CarWelding sky=188 257-395 m but ~0% px (label likely not sky-like in interiors); CountryHouse none.
- Unlabelled envs with sky not caught by saturation: OldTownNight has visible sky (clouds) at finite 150-218 m, 0 saturated px (stays valid depth under released rule AND v2 default).
- No env besides Antiquity3D has seg-sky finite <655 m share > 1.3% outside the near-dome/interior envs already listed (Apocalyptic 0.24%, AbandonedSchool 0.23%, ModUrbanCity 1.2% within its near dome class).
