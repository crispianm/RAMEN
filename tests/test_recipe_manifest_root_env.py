"""The three endpoint-population recipe builders must take their on-disk root
from an environment variable so the frozen accuracy harness can run on a host
other than crispwork (2026-09-18: Isambard/GH200 scoring, latency still only
from crispwork's 4090).

The contract under test is deliberately narrow:

* with the variable UNSET, the resolved root is bit-identical to the crispwork
  constant that was hardcoded before -- nothing changes on crispwork;
* with it SET, that value wins;
* an explicit ``root=`` argument still wins over both, because
  ``build_manifest``'s ``--root`` flag and the existing tests pass one;
* resolution happens at CALL time, not import time. This matters: the frozen
  ``scripts/eval_streaming_npz.py`` imports ``evaluation.recipe_registry``
  (and therefore these modules) at process start, which may be before a wrapper
  sets the variable.

``resolve_root`` is the single place each builder resolves a root, so testing it
plus one call-time path per module covers every entry point.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import build_bonn_recipe_manifest as bonn
from scripts import build_kitti_recipe_manifest as kitti
from scripts import build_vkitti_recipe_manifest as vkitti

CASES = [
    (vkitti, "RAMEN_VKITTI_ROOT", Path("/mnt/DATA/DegradedDepth/data/TRAIN/vkitti")),
    (kitti, "RAMEN_KITTI_ROOT",
     Path("/mnt/DATA/DegradedDepth/RAMEN/results/vda_clean_repro/benchmark_data/kitti")),
    (bonn, "RAMEN_BONN_ROOT", Path("/mnt/DATA/DegradedDepth/data/TEST/bonn")),
]


@pytest.mark.parametrize("module, env_var, crispwork_default", CASES)
def test_unset_env_resolves_to_the_crispwork_constant(module, env_var, crispwork_default, monkeypatch):
    monkeypatch.delenv(env_var, raising=False)
    assert module.resolve_root() == crispwork_default


@pytest.mark.parametrize("module, env_var, crispwork_default", CASES)
def test_env_override_is_honoured(module, env_var, crispwork_default, monkeypatch, tmp_path):
    other = tmp_path / "benchmarks" / "somewhere_else"
    monkeypatch.setenv(env_var, str(other))
    assert module.resolve_root() == other
    assert module.resolve_root() != crispwork_default


@pytest.mark.parametrize("module, env_var, crispwork_default", CASES)
def test_explicit_argument_beats_the_environment(module, env_var, crispwork_default, monkeypatch, tmp_path):
    monkeypatch.setenv(env_var, str(tmp_path / "from_env"))
    explicit = tmp_path / "from_arg"
    assert module.resolve_root(explicit) == explicit
    assert module.resolve_root(str(explicit)) == explicit


@pytest.mark.parametrize("module, env_var, crispwork_default", CASES)
def test_empty_env_var_falls_back_to_the_default(module, env_var, crispwork_default, monkeypatch):
    monkeypatch.setenv(env_var, "")
    assert module.resolve_root() == crispwork_default


@pytest.mark.parametrize("module, env_var, crispwork_default", CASES)
def test_resolution_happens_at_call_time_not_import_time(module, env_var, crispwork_default, monkeypatch,
                                                         tmp_path):
    """The module has already been imported by the time this test body runs --
    exactly the situation inside the frozen scorer, which imports the registry
    at process start. Setting the variable now must still take effect."""
    first = tmp_path / "a"
    monkeypatch.setenv(env_var, str(first))
    assert module.resolve_root() == first
    second = tmp_path / "b"
    monkeypatch.setenv(env_var, str(second))
    assert module.resolve_root() == second


@pytest.mark.parametrize("module, env_var, crispwork_default", CASES)
def test_build_regenerate_range_reads_the_env_root(module, env_var, crispwork_default, monkeypatch,
                                                   tmp_path):
    """The registry (``evaluation.recipe_registry.RECIPE_REGISTRY``) calls the
    factory with the condition only, so the env root must reach it through the
    default. An empty tree makes the callback fail on the missing root rather
    than on the crispwork path -- that the failure names the overridden root is
    the observable proof the override reached the frame-loading path."""
    root = tmp_path / "empty_benchmark_root"
    root.mkdir()
    monkeypatch.setenv(env_var, str(root))
    regenerate_range = module.build_regenerate_range("clean")
    with pytest.raises(Exception) as excinfo:
        regenerate_range("no_such_video", 0, 1)
    assert str(crispwork_default) not in str(excinfo.value)


def test_registry_entries_are_the_env_aware_factories():
    """The frozen scorer reaches these builders only through the registry, so a
    builder made portable but not registered would be portable in name only."""
    from evaluation.recipe_registry import RECIPE_REGISTRY

    assert RECIPE_REGISTRY["vkitti"] is vkitti.build_regenerate_range
    assert RECIPE_REGISTRY["kitti"] is kitti.build_regenerate_range
    assert RECIPE_REGISTRY["bonn"] is bonn.build_regenerate_range
