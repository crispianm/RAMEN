"""Extension point for a recipe-locked manifest's dataset-specific regeneration
callback (:class:`evaluation.alignment_free.RecipeManifest`'s ``regenerate_range``
argument, ``(video_id, start, end) -> rows``) -- same registry pattern as
``evaluation.model_registry``'s ``MODEL_REGISTRY``.

Deliberately NOT defined inside ``scripts/eval_streaming_npz.py`` itself, even
though that is where it is consumed: that file is one of the four frozen under
the ``benchmark_fidelity`` gate (``docs/BENCHMARK_PROTOCOL.md``'s ``BENCH-06``),
and a gate re-run cannot detect anything about a registry entry's correctness --
``RECIPE_REGISTRY`` is never consulted on the pixel-manifest (``FrameManifest``)
path the gate's own reference triples run through, so every entry added here is,
by construction, invisible to that check. Registering a per-benchmark builder
(Bonn, VKITTI, KITTI, ...) is therefore a plain, non-frozen commit -- reviewed and
covered by its own equivalence tests, not gated -- rather than triggering a gate
re-run whose result was already knowable in advance.

Keyed by ``--dataset`` alone (``dataset`` keeps meaning the dataset, matching
every other artifact field and §5 of ``docs/DEGRADED_EVAL_PROTOCOL.md``); the
factory takes ``--condition`` as its one argument rather than condition being
folded into the dataset key, so a builder's clean vs degraded
``regenerate_range`` share one registry entry. A pixel-materialised
``FrameManifest`` (:func:`evaluation.alignment_free.write_frame_manifest`)
never consults this at all.

Registers all four endpoint-population builders (docs/DEGRADED_EVAL_PROTOCOL.md
§3): ``"bonn"``, ``"vkitti"``, ``"kitti"``, ``"kitti_val"``.

CORRECTED 2026-08-31 (TOOL-13): this docstring previously claimed KITTI-val
(the single-frame supplement, no video structure) "is not built through this
recipe-locking mechanism, so it has no entry here." That was wrong --
``scripts/build_kitti_val_recipe_manifest.py`` writes its manifest through
the same :class:`evaluation.alignment_free.RecipeManifest` digest-only format
as the other three builders (confirmed directly against
``tests/test_build_kitti_val_recipe_manifest.py``), so its manifest cannot be
read back without a ``regenerate_range`` callback either -- ``scripts/
eval_streaming_npz.py --dataset kitti_val`` raised ``ValueError: ... is a
recipe manifest; regenerate_range must be supplied`` with no entry present.
The builder's own ``build_regenerate_range`` already existed and needed only
registering here, exactly as TOOL-13 diagnosed before this fix.
"""

from __future__ import annotations

from typing import Callable

from scripts.build_bonn_recipe_manifest import build_regenerate_range as _bonn_regenerate_range
from scripts.build_kitti_recipe_manifest import build_regenerate_range as _kitti_regenerate_range
from scripts.build_kitti_val_recipe_manifest import build_regenerate_range as _kitti_val_regenerate_range
from scripts.build_vkitti_recipe_manifest import build_regenerate_range as _vkitti_regenerate_range

RECIPE_REGISTRY: dict[str, Callable[[str], Callable[[str, int, int], list]]] = {
    "bonn": _bonn_regenerate_range,
    "vkitti": _vkitti_regenerate_range,
    "kitti": _kitti_regenerate_range,
    "kitti_val": _kitti_val_regenerate_range,
}
