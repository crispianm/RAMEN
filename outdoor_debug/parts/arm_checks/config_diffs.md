# Resolved-config diffs (flat keys; yaml written to configs/arm_<name>.yaml)

A0 == effective_c3_seed45.yaml (no override): True


## A0 -> A0b  [OK]  (1 differing keys)
- data.seed: 45 -> 46

## A0 -> TS2  [OK]  (2 differing keys)
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true

## TS2 -> TS2_H80  [OK]  (2 differing keys)
- loss.sky_hinge_min_depth: <absent> -> 80.0
- loss.sky_hinge_weight: 0.0 -> 0.1

## TS2 -> TS2_H300  [OK]  (2 differing keys)
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1

## TS2 -> TS2_VK655  [OK]  (2 differing keys)
- data.sources[vkitti].vkitti_max_depth: 655 -> 656
- data.sources[vkitti].vkitti_sky_as_valid_depth: <absent> -> true

## A0 -> TS2_H80  [info]  (4 differing keys)
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- loss.sky_hinge_min_depth: <absent> -> 80.0
- loss.sky_hinge_weight: 0.0 -> 0.1

## A0 -> TS2_H300  [info]  (4 differing keys)
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- loss.sky_hinge_min_depth: <absent> -> 300.0
- loss.sky_hinge_weight: 0.0 -> 0.1

## A0 -> TS2_VK655  [info]  (4 differing keys)
- data.sources[tartanair].ts2_seg_root: <absent> -> "/projects/b5dh/data/train/tartanair_seg/"
- data.sources[tartanair].ts2_sky: <absent> -> true
- data.sources[vkitti].vkitti_max_depth: 655 -> 656
- data.sources[vkitti].vkitti_sky_as_valid_depth: <absent> -> true

ALL EXPECTED DIFFS EXACT: True
