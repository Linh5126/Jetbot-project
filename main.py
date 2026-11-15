"""MOT17 benchmark CLI. Run from the project root with the dataset installed."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data.mot17 import discover_sequences
from src.evaluation.benchmark_common import load_config
from src.evaluation.benchmark_legacy import run_sequence
from src.evaluation.benchmark_v5 import run_v5_experiment
from src.identity.manager_v5 import IdentityManagerV5
from src.reid.model_config import REID_CHECKPOINTS, default_output_root, select_reid_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JetBot MOT17 independent evaluation: Baseline, V1-V5.1"
    )
    parser.add_argument("--mode", choices=["baseline", "v1", "v2", "v3", "v4", "v5", "v5_1"])
    parser.add_argument("--sequence", help="Ví dụ: MOT17-02-FRCNN")
    parser.add_argument("--all-sequences", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Chỉ chạy N frame để kiểm tra nhanh")
    parser.add_argument(
        "--ablation",
        choices=["all", "appearance_only", "motion_only", "fixed_fusion", "adaptive_fusion"],
        default="all",
        help="V5/V5.1 ablation strategy; all runs four strategies with shared detections/embeddings",
    )
    parser.add_argument("--save-video", action="store_true",
                        help="Lưu video annotate cho tất cả phiên bản (tăng dung lượng đầu ra)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--reid-model", choices=tuple(REID_CHECKPOINTS), default=None,
                        help="V5/V5.1: OSNet pretrained MSMT17 (x0.5 or x1.0)")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Write a separate experiment tree without overwriting original outputs")
    parser.add_argument(
        "--detector-confidence", type=float, default=None,
        help="Override detector.confidence for a controlled low-confidence ByteTrack experiment",
    )
    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config(Path(args.config))
    if args.detector_confidence is not None:
        if not 0.0 <= args.detector_confidence <= 1.0:
            parser.error("--detector-confidence must be in [0,1]")
        cfg["detector"]["confidence"] = args.detector_confidence
    mode = args.mode or cfg["experiment"]["mode"]
    if args.reid_model is not None and mode not in {"v5", "v5_1"}:
        parser.error("--reid-model requires --mode v5 or v5_1")
    cfg = select_reid_model(cfg, args.reid_model)

    root = Path(cfg["dataset"]["root"])
    sequences = discover_sequences(root, cfg["dataset"]["split"],
                                   cfg["dataset"].get("sequence_suffix", "FRCNN"))
    if args.sequence:
        sequences = [s for s in sequences if s.name == args.sequence]
        if not sequences:
            raise SystemExit(f"Không tìm thấy sequence {args.sequence} trong {root}")
    elif not args.all_sequences:
        preferred = cfg["dataset"]["default_sequence"]
        sequences = [s for s in sequences if s.name == preferred] or sequences[:1]

    if not sequences:
        raise SystemExit(
            f"Không tìm thấy MOT17. Kiểm tra cấu trúc: {root}/train/MOT17-XX-FRCNN/img1"
        )

    if args.output_root:
        output_root = args.output_root
    elif mode in {"v5", "v5_1"}:
        output_root = default_output_root(
            Path(cfg["output"]["root"]), cfg["reid"]["v5"]["model_name"]
        )
    else:
        output_root = Path(cfg["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)

    if mode in {"v5", "v5_1"}:
        strategies = (
            list(IdentityManagerV5.STRATEGIES)
            if args.ablation == "all"
            else [args.ablation]
        )
        # Stable, readable experiment order for reports and CSV files.
        order = ["appearance_only", "motion_only", "fixed_fusion", "adaptive_fusion"]
        strategies = [s for s in order if s in strategies]
        result = run_v5_experiment(
            sequences=sequences,
            cfg=cfg,
            output_root=output_root,
            strategies=strategies,
            max_frames=args.max_frames,
            save_video=args.save_video,
            experiment_name=mode,
            quality_aware=(mode == "v5_1"),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    summaries = []
    for seq in sequences:
        summaries.append(run_sequence(
            seq, mode, cfg, output_root, args.max_frames, save_video=args.save_video
        ))

    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "run_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
