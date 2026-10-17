# TartanAir per-env sky audit (local TRAIN envs, 20 frames/env spread across trajectories)

Data: official tartanairpy 1.4.0 seg_lcam_front (Data_easy), sha256 in SEG_SHA256SUMS; depth from the local TRAIN tree (read only). Label env names: **TE = training environment, not held out** (Downtown, Hospital, ForestEnv, OldBrickHouseNight); all other rows are also local TRAIN envs. G = in tartanairpy GLASS_PROBLEM_ENVS (random_accessor.py L145-147). Sky = seg class named sky/skysphere/sky0/cloud in the env's seg_label_map.json. 'rule' = released loader sky rule `~isfinite(d) | d>1e4`.

| env | flags | sky label (id) | sky % px | stored sky depth min / mode / max (m) | seg-sky caught by rule % | seg-sky finite <655 m % | seg-sky 655-1e4 m % | non-sky px flagged by rule (% of px; top classes) |
|---|---|---|---|---|---|---|---|---|
| AbandonedFactory | - | sky=160 | 20.385 | 9440 / 11616 / 16400 | 97.9 | 0.00 | 2.12 | 0.0  |
| AmericanDiner | - | sky=0,sky0=242 | 0.001 | 65504 / 65504 / 65504 | 100.0 | 0.00 | 0.00 | 0.0  |
| AmusementPark | - | sky=28 | 27.559 | 222 / 300 / 434 | 0.0 | 100.00 | 0.00 | 0.0  |
| ArchVizTinyHouseDay | - | sky=229 | 2.509 | 9 / 11632 / 16344 | 97.4 | 0.07 | 2.55 | 0.0  |
| ArchVizTinyHouseNight | - | skysphere=36 | 0.0 | - | - | - | - | 2.446 id118:100% |
| BrushifyMoon | - | sky=59 | 42.522 | 65504 / 65504 / 65504 | 100.0 | 0.00 | 0.00 | 0.075 earth:87%; sun:13% |
| CoalMine | - | none | 0.0 | - | - | - | - | 0.0  |
| ConstructionSite | - | sky=119 | 14.231 | 65504 / 65504 / 65504 | 100.0 | 0.00 | 0.00 | 0.013 ground:100% |
| DesertGasStation | - | sky=214 | 25.618 | 34560 / 42560 / 60096 | 100.0 | 0.00 | 0.00 | 0.171 ground:100% |
| Downtown | TE | sky=208 | 5.547 | 233 / 332 / 510 | 0.0 | 100.00 | 0.00 | 0.0  |
| ForestEnv | TE | sky=27 | 20.728 | 10 / 11616 / 16448 | 97.9 | 0.00 | 2.05 | 0.0 ground:100% |
| GreatMarsh | - | sky=23 | 29.833 | 2 / 8208 / 10152 | 0.1 | 0.35 | 99.50 | 0.0  |
| HongKong | - | none | 0.0 | - | - | - | - | 0.205 id0:100% |
| Hospital | TE | sky=137 | 0.01 | 31 / 3210 / 3570 | 0.0 | 0.12 | 99.88 | 0.0  |
| JapaneseCity | - | sky=217 | 3.187 | 9448 / 11568 / 16376 | 96.6 | 0.00 | 3.37 | 0.0  |
| ModernCityDowntown | - | sky=123 | 6.964 | 9448 / 11584 / 16400 | 98.3 | 0.00 | 1.71 | 0.0  |
| NordicHarbor | G | sky=8 | 12.995 | 9400 / 11584 / 16328 | 98.0 | 0.00 | 2.00 | 0.0  |
| Ocean | - | none | 0.0 | - | - | - | - | 1.038 z:100%; ground:0% |
| Office | G | skysphere=182 | 0.486 | 9456 / 12064 / 15528 | 96.4 | 0.00 | 3.60 | 0.0  |
| OldBrickHouseNight | TE G | none | 0.0 | - | - | - | - | 1.831 id0:100% |
| PolarSciFi | - | sky=9 | 26.105 | 9456 / 11608 / 16416 | 98.5 | 0.00 | 1.53 | 0.0  |
| Prison | G | sky=109 | 23.611 | 36 / 87 / 177 | 0.0 | 100.00 | 0.00 | 0.0  |
| Restaurant | - | sky=70 | 1.899 | 9464 / 11560 / 16352 | 97.5 | 0.00 | 2.54 | 0.0  |
| Rome | - | sky=109 | 3.704 | 9448 / 11592 / 15416 | 95.1 | 0.00 | 4.86 | 0.0  |
| SeasonalForestWinter | - | sky=64 | 33.994 | 21 / 11640 / 16464 | 98.3 | 0.00 | 1.71 | 0.0  |
| ShoreCaves | - | sky=64 | 34.188 | 91 / 262 / 498 | 0.0 | 100.00 | 0.00 | 0.0  |
| Slaughter | G | sky=70,skysphere=143 | 9.168 | 9432 / 11584 / 16376 | 98.4 | 0.00 | 1.57 | 0.0  |
| WesternDesertTown | G | sky=29 | 8.026 | 2 / 11608 / 16384 | 98.0 | 0.02 | 1.93 | 0.0  |

## Findings (20 frames/env; small shares such as Hospital 0.01% sky or Office 0.5% are low-count)

1. **Six labelled envs: the d>1e4 rule catches ~0% of seg-sky** because the sky is a real finite dome closer than 10 km: Downtown (233-510 m, mode 332 m), AmusementPark (222-434 m), Prison (36-177 m), ShoreCaves (91-498 m), Hospital (31-3570 m, 99.9% in 655-1e4), GreatMarsh (mostly 5.9-10.2 km; 0.15% caught). In the released loader those pixels are sky=0 and mask=1, i.e. valid metric depth of 36 m-10 km. Downtown/Hospital/ForestEnv/OldBrickHouseNight are training environments, not held out.
2. **Thirteen labelled envs: rule catches 95.1-98.5%**; the dome spans 9.4-16.4 km (mode ~11.6 km), so 1.5-4.9% of seg-sky pixels sit at 9.4-10.0 km: sky=0 and mask=1 (valid < max_depth 10000). AbandonedFactory, ArchVizTinyHouseDay, ForestEnv, JapaneseCity, ModernCityDowntown, NordicHarbor, Office, PolarSciFi, Restaurant, Rome, SeasonalForestWinter, Slaughter, WesternDesertTown.
3. **Four labelled envs 100% caught**: AmericanDiner (65504; only 0.001% of px, and its `sky` id is 0, the same id that HongKong/OldBrickHouseNight use for unlabelled saturated px), BrushifyMoon (65504, 42.5% of px), ConstructionSite (65504, 14.2%), DesertGasStation (dome 34-60 km, 25.6%). The 65504 sentinel is stored only in AmericanDiner, BrushifyMoon, ConstructionSite, HongKong, OldBrickHouseNight (reference frames in parity_reference.json: n_65504 > 0).
4. **Four envs have no sky label** (CoalMine, HongKong, Ocean, OldBrickHouseNight) and **ArchVizTinyHouseNight** has `skysphere=36` that never appears in 20 frames (its saturated px, 2.4% of px, carry id 118, which is not in name_map). Saturated px: HongKong 0.21% (class id 0), OldBrickHouseNight 1.83% (id 0), Ocean 1.04% (class `z`=27), CoalMine 0. These are the cases where seg gives no sky information; saturation-only is the only signal.
5. **Glass-problem envs** (local: NordicHarbor, Slaughter, Prison, Office, WesternDesertTown, OldBrickHouseNight): no glass-specific saturation artefact in the sample (non-sky px flagged by the rule = 0.0% in NordicHarbor, Slaughter, Prison, Office, WesternDesertTown); Prison is a sky-dome-near env (item 1); OldBrickHouseNight is item 4. Not a sky-rule effect in this sample.
6. **Trees-as-sky (tartanair_tools #46)**: not reproduced; among non-sky px flagged by the rule, tree/foliage/bush/plant/grass-like classes are 0.0% in all 28 envs (20 frames each). Non-sky saturated px are tiny: BrushifyMoon 0.075% (earth 87%, sun 13%), ConstructionSite 0.013% (ground), DesertGasStation 0.171% (ground), ArchVizTinyHouseNight 2.4%, HongKong 0.21%, OldBrickHouseNight 1.83%, Ocean 1.04% (all unmapped/unlabelled classes).
7. Seg-sky px with finite depth < 655 m outside the near-dome envs are 0-0.35% (label-edge/aliasing): ForestEnv 0.001%, GreatMarsh 0.35%, ArchVizTinyHouseDay 0.07%, WesternDesertTown 0.02%.

Flagged rule for envs with no usable sky label: saturated (`~isfinite(d) | d>=1e4`) px are UNCERTAIN (sky=0, mask=0) by default; opt-in flag `fallback_sat_is_sky` labels them sky, only after a per-env visual check (not done here).
