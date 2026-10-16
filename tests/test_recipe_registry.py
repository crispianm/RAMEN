from evaluation.recipe_registry import RECIPE_REGISTRY
from scripts.build_kitti_val_recipe_manifest import KITTI_VAL_ROOT


def test_recipe_registry_has_exactly_the_four_endpoint_builders():
    """CORRECTED 2026-08-31 (TOOL-13): KITTI-val's own manifest is written
    through the same RecipeManifest digest-only format as the other three
    (confirmed against tests/test_build_kitti_val_recipe_manifest.py) and
    cannot be read back without a registered regenerate_range -- it is not
    exempt from this mechanism, contrary to what this test and
    evaluation/recipe_registry.py's own docstring previously claimed."""
    assert set(RECIPE_REGISTRY) == {"bonn", "vkitti", "kitti", "kitti_val"}


def test_recipe_registry_entries_are_condition_keyed_factories():
    """kitti_val's own build_regenerate_range (unlike the other three builders,
    which are lazy -- they touch disk only inside the returned closure) reads
    its real data root eagerly at factory-call time, so this smoke check is
    skipped for it when that root isn't present on this host (only crispwork
    has it) -- exercised for real as part of the KITTI-val supplement run
    itself, not duplicated here against a host that lacks the data."""
    for dataset, factory in RECIPE_REGISTRY.items():
        if dataset == "kitti_val" and not KITTI_VAL_ROOT.exists():
            continue
        regenerate_range = factory("clean")
        assert callable(regenerate_range)


def test_recipe_registry_accepts_a_condition_keyed_factory():
    """The contract eval_streaming_npz.py's main() relies on: a zero-arg entry per
    --dataset, whose factory takes --condition and returns a regenerate_range
    callable. Exercised here without depending on any real builder."""
    RECIPE_REGISTRY["_test_dataset"] = lambda condition: (
        lambda video_id, start, end: [(condition, video_id, start, end)]
    )
    try:
        regenerate_range = RECIPE_REGISTRY["_test_dataset"]("degraded")
        assert regenerate_range("v0", 0, 2) == [("degraded", "v0", 0, 2)]
    finally:
        del RECIPE_REGISTRY["_test_dataset"]
