"""D3 data package — clip-batch contract for degraded metric video depth."""

from __future__ import annotations

import os

# IRS depth maps are OpenEXR; OpenCV disables the EXR codec by default (>=4.5). Enable it
# BEFORE cv2 is first imported by any submodule below.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from .video_clip_dataset import VideoDepthClipDataset
from .vkitti_dataset import VKITTIDataset
from .pseudo_gt_dataset import PseudoGTVideoDataset
from .irs_dataset import IRSDataset
from .pointodyssey_dataset import PointOdysseyDataset
from .nyu_dataset import NYUDataset
from .dark_zurich_dataset import DarkZurichDataset
from .scannetpp_dataset import ScanNetPPDataset
from .video_only_dataset import VideoOnlyDataset

__all__ = ["VideoDepthClipDataset", "VKITTIDataset", "PseudoGTVideoDataset",
           "IRSDataset", "PointOdysseyDataset", "NYUDataset", "DarkZurichDataset",
           "ScanNetPPDataset", "VideoOnlyDataset"]
