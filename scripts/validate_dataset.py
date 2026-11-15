"""Validate MOT17 sequence folders, frame images and pedestrian GT files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.data.mot17 import discover_sequences, read_mot_gt


def main():
    parser = argparse.ArgumentParser(description="Validate the configured MOT17 dataset.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--strict-seven", action="store_true",
                        help="Require exactly seven FRCNN sequences.")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    data_cfg = cfg["dataset"]
    dataset_root = Path(data_cfg["root"]).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    sequences = discover_sequences(
        dataset_root, data_cfg.get("split", "train"), data_cfg.get("sequence_suffix", "FRCNN")
    )
    problems = []
    print("Dataset root:", dataset_root.resolve())
    print("Found sequences:", len(sequences))
    for seq in sequences:
        images = sorted(list((seq / "img1").glob("*.jpg")) + list((seq / "img1").glob("*.png")))
        gt_path = seq / "gt" / "gt.txt"
        gt_rows = read_mot_gt(gt_path)
        status = "OK" if images and gt_path.is_file() and gt_rows else "INVALID"
        print("%-22s | images=%5d | GT rows=%6d | %s" % (seq.name, len(images), len(gt_rows), status))
        if status != "OK":
            problems.append(seq.name)
    if args.strict_seven and len(sequences) != 7:
        problems.append("expected exactly 7 FRCNN sequences")
    if not sequences:
        problems.append("no matching sequences found")
    if problems:
        print("Validation failed:", "; ".join(problems))
        return 1
    print("Dataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
