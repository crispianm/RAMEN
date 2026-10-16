"""Ultrareview: VKITTI2 Scene01 / Scene20 are re-renders of KITTI-raw drives 2011_09_26_drive_0009 / 2011_10_03_drive_0047, which are in the KITTI DEV / KITTI-C eval sets.
The scene split never looked at that overlap. Default behaviour (hence every trained V2/F1/FV2 model) is unchanged; training now warns, and an opt-in excludes them."""
from __future__ import annotations

import warnings

import pytest

from data.vkitti_dataset import KITTI_EVAL_OVERLAP_SCENES, VKITTIDataset
from tests.test_vda_semantics_dataset import COMMON, _write_16bit_depth, _write_rgb
from train import _make_source

SCENES = ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20")


@pytest.fixture()
def root(tmp_path):
    for sc in SCENES:
        r = tmp_path / sc / "clone" / "frames" / "rgb" / "Camera_0"
        dp = tmp_path / sc / "clone" / "frames" / "depth" / "Camera_0"
        r.mkdir(parents=True)
        dp.mkdir(parents=True)
        for i in range(3):
            _write_rgb(r / f"rgb_{i:05d}.jpg")
            _write_16bit_depth(dp / f"depth_{i:05d}.png")
    return tmp_path


def _scenes_of(ds):
    return {c[2].split("/")[0] for c in ds.clips}


def test_overlap_table_names_the_two_scenes():
    assert KITTI_EVAL_OVERLAP_SCENES == {"Scene01": "2011_09_26_drive_0009_sync", "Scene20": "2011_10_03_drive_0047_sync"}


def test_training_on_an_overlapping_scene_warns_and_changes_nothing_else(root):
    with pytest.warns(UserWarning, match="Scene20"):
        ds = VKITTIDataset(root=str(root), split="train", val_fraction=0.2, split_seed=43, **{k: v for k, v in COMMON.items() if k != "seed"}, seed=42)
    assert ds.eval_overlap_scenes == sorted(_scenes_of(ds) & set(KITTI_EVAL_OVERLAP_SCENES))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # val never warns
        VKITTIDataset(root=str(root), split="val", val_fraction=0.2, split_seed=43, **COMMON)


def test_exclude_eval_overlap_removes_both_scenes_and_keeps_train_val_disjoint(root):
    kw = dict(root=str(root), val_fraction=0.34, split_seed=43, exclude_eval_overlap=True, **COMMON)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # nothing left to warn about
        tr, va = VKITTIDataset(split="train", **kw), VKITTIDataset(split="val", **kw)
    assert not (_scenes_of(tr) | _scenes_of(va)) & set(KITTI_EVAL_OVERLAP_SCENES)
    assert _scenes_of(tr).isdisjoint(_scenes_of(va)) and _scenes_of(tr) | _scenes_of(va) == {"Scene02", "Scene06", "Scene18"}


def test_default_is_byte_identical_to_the_old_call(root):
    a = VKITTIDataset(root=str(root), split="train", val_fraction=0.2, split_seed=43, **COMMON)
    b = VKITTIDataset(root=str(root), split="train", val_fraction=0.2, split_seed=43, exclude_eval_overlap=False, **COMMON)
    assert [c[2:] for c in a.clips] == [c[2:] for c in b.clips] and a.clips                  # the scene list, hence the shuffle and every V2/F1/FV2 training split, is untouched


@pytest.mark.parametrize("flag", [None, False, True])
def test_multisource_vkitti_spec_key_reaches_the_dataset(root, flag):
    spec = {"type": "vkitti", "root": str(root), "variants": ["clone"], "val_fraction": 0.2}
    if flag is not None:
        spec["exclude_eval_overlap"] = flag
    common = dict(COMMON, degrade=False, resize_mode=None, split_seed=43, max_clips_per_video=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = _make_source(spec, "train", common)
    seen = _scenes_of(ds) & set(KITTI_EVAL_OVERLAP_SCENES)
    assert (not seen) if flag else bool(seen)                                              # 4 of the 5 scenes are train at val_fraction 0.2: at least one overlap scene is in
