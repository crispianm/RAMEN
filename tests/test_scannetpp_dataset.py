"""Pure byte-offset parsing logic for ScanNet++'s depth.bin format. No real data/GPU."""

from __future__ import annotations

from pathlib import Path

from data.scannetpp_dataset import _index_depth_bin


def _write_depth_bin(path: Path, frame_sizes: list[int]) -> None:
    with open(path, "wb") as f:
        for n in frame_sizes:
            f.write(n.to_bytes(4, "little"))
            f.write(b"\x00" * n)


def test_index_depth_bin_returns_offset_and_size_per_frame(tmp_path):
    path = tmp_path / "depth.bin"
    _write_depth_bin(path, [10, 0, 5])
    spans = _index_depth_bin(path)
    assert spans == [(4, 10), (18, 0), (22, 5)]


def test_index_depth_bin_empty_file(tmp_path):
    path = tmp_path / "depth.bin"
    path.write_bytes(b"")
    assert _index_depth_bin(path) == []
