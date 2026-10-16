from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image
from torch.utils.data import WeightedRandomSampler

from train import _build_multisource, load_config


def _write_tartanair_env(root, env_name, n_frames):
    """3 identically-sized trajectories: VideoDepthClipDataset reserves max(1, ...)
    trajectories for val even at val_fraction=0.0, chosen via a seeded *random* shuffle
    (not directory order) -- so with same-sized trajectories, which one lands in val doesn't
    matter, and train deterministically gets (3-1)*n_frames clips regardless of shuffle."""
    depth = np.full((64, 64), 5.0, dtype=np.float32)
    bgra = depth.reshape(64, 64, 1).view(np.uint8).reshape(64, 64, 4)
    for traj in ("P000", "P001", "P002"):
        img_dir = root / env_name / "Data_easy" / traj / "image_lcam_front"
        dep_dir = root / env_name / "Data_easy" / traj / "depth_lcam_front"
        img_dir.mkdir(parents=True)
        dep_dir.mkdir(parents=True)
        for i in range(n_frames):
            Image.new("RGB", (64, 64)).save(img_dir / f"{i:06d}_lcam_front.png")
            cv2.imwrite(str(dep_dir / f"{i:06d}_lcam_front.png"), bgra)


def _base_cfg(root_a, root_b, sampling, frame_counts):
    return {
        "data": {
            "num_frames": 1,
            "clip_stride": 1,
            "frame_stride": 1,
            "image_size": (64, 64),
            "sampling": sampling,
            "sources": [
                {"type": "tartanair", "root": str(root_a), "val_fraction": 0.0,
                 "frame_count": frame_counts[0]},
                {"type": "tartanair", "root": str(root_b), "val_fraction": 0.0,
                 "frame_count": frame_counts[1]},
            ],
        },
        "training": {"batch_size": 1},
    }


def test_proportional_sampling_total_mass_matches_declared_frame_count_ratio(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_tartanair_env(root_a, "EnvA", 5)
    _write_tartanair_env(root_b, "EnvB", 8)
    # Declared frame_count intentionally unrelated to the tiny fixture's own clip counts --
    # proportional sampling must follow the DECLARED numbers, not len(source).
    cfg = _base_cfg(root_a, root_b, "proportional", frame_counts=[897798, 21260])

    loader, _ = _build_multisource(cfg, "train", world_size=1, rank=0)
    assert isinstance(loader.sampler, WeightedRandomSampler)
    weights = loader.sampler.weights.tolist()
    n_a = 2 * 5   # 2 of 3 same-sized trajectories survive the val split; clips == frames here
    n_b = 2 * 8
    assert len(weights) == n_a + n_b
    mass_a = sum(weights[:n_a])
    mass_b = sum(weights[n_a:])
    assert mass_a == pytest.approx(897798, rel=1e-6)
    assert mass_b == pytest.approx(21260, rel=1e-6)
    # uniform within each source
    assert len(set(round(w, 6) for w in weights[:n_a])) == 1
    assert len(set(round(w, 6) for w in weights[n_a:])) == 1


def test_proportional_sampling_requires_frame_count(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_tartanair_env(root_a, "EnvA", 3)
    _write_tartanair_env(root_b, "EnvB", 3)
    cfg = _base_cfg(root_a, root_b, "proportional", frame_counts=[100, 100])
    del cfg["data"]["sources"][1]["frame_count"]

    with pytest.raises(ValueError, match="frame_count"):
        _build_multisource(cfg, "train", world_size=1, rank=0)


def test_equal_sampling_is_unaffected_by_declared_frame_count(tmp_path):
    """Backward compatibility: existing configs (sampling unset, or 'equal') must keep
    every source equally represented regardless of size -- the pre-existing behaviour."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_tartanair_env(root_a, "EnvA", 3)
    _write_tartanair_env(root_b, "EnvB", 9)
    cfg = _base_cfg(root_a, root_b, "equal", frame_counts=[100, 100])

    loader, _ = _build_multisource(cfg, "train", world_size=1, rank=0)
    weights = loader.sampler.weights.tolist()
    mass_a = sum(weights[:2 * 3])
    mass_b = sum(weights[2 * 3:])
    assert mass_a == pytest.approx(mass_b, rel=1e-6)


def test_sampling_unset_defaults_to_equal_matching_legacy_uniform_sampler_default(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_tartanair_env(root_a, "EnvA", 2)
    _write_tartanair_env(root_b, "EnvB", 6)
    cfg = _base_cfg(root_a, root_b, sampling=None, frame_counts=[100, 100])
    del cfg["data"]["sampling"]

    loader, _ = _build_multisource(cfg, "train", world_size=1, rank=0)
    assert isinstance(loader.sampler, WeightedRandomSampler)
    weights = loader.sampler.weights.tolist()
    mass_a = sum(weights[:2 * 2])
    mass_b = sum(weights[2 * 2:])
    assert mass_a == pytest.approx(mass_b, rel=1e-6)


def _write_config(tmp_path, sources):
    cfg = {
        "data": {"sampling": "proportional", "sources": sources},
        "training": {"batch_size": 1},
    }
    path = tmp_path / "cfg.yaml"
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_load_config_accepts_frame_count_matching_the_hashed_inventory(tmp_path):
    """Opus's hardening request, 2026-08-27: proportional sampling must cross-check its
    declared frame_count against provenance/inventory/val01_dataset_inventory_20260827.json,
    not just trust the config's own hand-typed number."""
    sources = [
        {"type": "tartanair", "root": "/x", "frame_count": 897798},
        {"type": "vkitti", "root": "/y", "frame_count": 2126},
    ]
    cfg = load_config(_write_config(tmp_path, sources))
    assert cfg["data"]["sources"][0]["frame_count"] == 897798


def test_load_config_rejects_frame_count_mismatched_against_the_hashed_inventory(tmp_path):
    sources = [
        {"type": "tartanair", "root": "/x", "frame_count": 897798},
        {"type": "vkitti", "root": "/y", "frame_count": 999999},   # wrong -- real value is 2126
    ]
    with pytest.raises(ValueError, match="does not match|records"):
        load_config(_write_config(tmp_path, sources))


def test_load_config_skips_the_frame_count_check_when_sampling_is_not_proportional(tmp_path):
    """Backward compat: existing configs (sampling unset/'equal') must never fail to load
    just because they don't declare frame_count at all."""
    path = tmp_path / "cfg.yaml"
    with path.open("w") as f:
        yaml.safe_dump({"data": {"sources": [{"type": "tartanair", "root": "/x"}]}}, f)
    cfg = load_config(path)   # must not raise
    assert cfg["data"]["sources"][0]["type"] == "tartanair"


def test_frame_count_inventory_artifact_is_valid_json_with_the_expected_keys():
    from train import _FRAME_COUNT_INVENTORY
    with _FRAME_COUNT_INVENTORY.open() as f:
        d = json.load(f)
    assert set(d["frame_counts"]) == {"tartanair", "vkitti", "irs", "pointodyssey"}
    assert all(isinstance(v, int) and v > 0 for v in d["frame_counts"].values())



def test_ddp_ranks_keep_the_weighted_sampler_with_distinct_streams(tmp_path):
    """Multi-GPU runs must keep the recipe's per-source balance: every rank gets a
    WeightedRandomSampler over the SAME weights, 1/world_size of an epoch, its own draws."""
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _write_tartanair_env(root_a, "EnvA", 5)
    _write_tartanair_env(root_b, "EnvB", 8)
    cfg = _base_cfg(root_a, root_b, "proportional", frame_counts=[897798, 21260])

    single, _ = _build_multisource(cfg, "train", world_size=1, rank=0)
    r0, _ = _build_multisource(cfg, "train", world_size=2, rank=0)
    r1, _ = _build_multisource(cfg, "train", world_size=2, rank=1)
    for r in (r0, r1):
        assert isinstance(r.sampler, WeightedRandomSampler)
        assert r.sampler.weights.tolist() == single.sampler.weights.tolist()
        assert r.sampler.num_samples == max(1, single.sampler.num_samples // 2)
    assert r0.sampler.generator.initial_seed() != r1.sampler.generator.initial_seed()
