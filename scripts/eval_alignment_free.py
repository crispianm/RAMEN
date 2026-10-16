#!/usr/bin/env python3
"""Evaluate a prediction cache written by scripts/dump_predictions.py (Tier 1)."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.alignment_free import DepthEvaluator, EvaluationMetadata, read_prediction_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", help="NPZ written by scripts/dump_predictions.py "
                                   "(video_id, shape, pred, gt, mask)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()
    evaluator = DepthEvaluator(EvaluationMetadata(args.model, args.dataset, args.split))
    for vid, pred, gt, mask in read_prediction_cache(args.cache):
        evaluator.add_frame(vid, pred, gt, mask)
    result = evaluator.write_json(args.out)
    print(json.dumps({"out": str(Path(args.out)), "counts": result["counts"],
                      "raw": result["metrics"]["raw"]["frame_macro"]}, indent=2))


if __name__ == "__main__":
    main()
