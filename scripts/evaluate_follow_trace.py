"""Score a manually selected PC-follow trace against aligned MOT17 train GT.

Example:
python scripts/evaluate_follow_trace.py --trace outputs/demo_11_v24_trace.csv \
    --sequence MOT17-11-FRCNN
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.follow_trace import GroundTruthIndex, evaluate_follow_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frame-wise PC following with MOT17 GT")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sequence", required=True,
                        help="MOT17-11-FRCNN or path to its train sequence folder")
    parser.add_argument("--gt-root", type=Path, default=Path("data/datasets/MOT17/train"))
    parser.add_argument("--frame-offset", type=int, default=0,
                        help="Video-to-GT frame index offset (default 0)")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-visibility", type=float, default=0.20,
                        help="Same GT visibility threshold as during trace recording")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sequence_path = Path(args.sequence)
    if not (sequence_path / "gt" / "gt.txt").exists():
        sequence_path = args.gt_root / args.sequence
    gt = GroundTruthIndex(sequence_path, min_visibility=args.min_visibility)
    metrics = evaluate_follow_trace(args.trace, gt, iou_threshold=args.iou,
                                    frame_offset=args.frame_offset)
    output = args.output or args.trace.with_suffix(".metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items()
                      if key not in ("sessions",)}, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
