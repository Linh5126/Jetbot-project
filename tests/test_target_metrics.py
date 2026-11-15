"""Tiny hand-checkable MOT17 examples: no model weights or datasets required."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.evaluation.target_metrics import aggregate_target_centric, evaluate_target_centric


def write_case(tmp_path: Path, gt_rows: list[tuple], pred_rows: list[tuple]):
    gt = tmp_path / "gt.txt"
    gt.write_text("\n".join(",".join(map(str, (f, gid, x, 0, 10, 10, 1, 1, 1)))
                            for f, gid, x in gt_rows), encoding="utf-8")
    pred = tmp_path / "identity_log.csv"
    with pred.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "target_id", "x1", "y1", "x2", "y2"])
        writer.writerows((f, tid, x, 0, x + 10, 10) for f, tid, x in pred_rows)
    return gt, pred


def target(result, gt_id):
    return next(row for row in result["target_details"] if row["gt_person_id"] == gt_id)


def test_wrong_follow_is_not_target_id_mismatch(tmp_path):
    # GT1 exits at frame 2; its locked ID 10 incorrectly follows GT2.
    # At frame 3 GT1 recovers under ID 10.
    gt, pred = write_case(tmp_path,
        [(1, 1, 0), (1, 2, 30), (2, 2, 30), (3, 1, 0), (3, 2, 30)],
        [(1, 10, 0), (1, 20, 30), (2, 10, 30), (3, 10, 0), (3, 20, 30)])
    result = evaluate_target_centric(gt, pred, lost_gap_frames=1, max_reacquisition_frames=5)
    one = target(result, 1)
    assert one["wrong_identity_frames"] == 0
    assert one["locked_id_wrong_person_frames"] == 1
    assert one["locked_id_wrong_person_rate"] == pytest.approx(1 / 3)
    assert one["reacquisition_episodes"] == 1
    assert one["successful_reacquisitions"] == 1
    assert one["wrong_person_reacquisitions"] == 1
    assert one["recovery_latencies_frames"] == [2]


def test_terminal_exit_is_right_censored_even_after_deadline(tmp_path):
    gt, pred = write_case(tmp_path,
        [(1, 1, 0), (2, 1, 0)] + [(f, 2, 30) for f in range(1, 12)],
        [(1, 10, 0), (2, 10, 0)] + [(f, 20, 30) for f in range(1, 12)])
    one = target(evaluate_target_centric(gt, pred, lost_gap_frames=2,
                                          max_reacquisition_frames=3), 1)
    assert one["reacquisition_episodes"] == 0
    assert one["failed_reacquisitions"] == 0
    assert one["right_censored_episodes"] == 1
    assert one["reacquisition_success_rate"] is None


def test_unrecovered_visible_gt_counts_failure(tmp_path):
    gt, pred = write_case(tmp_path,
        [(f, 1, 0) for f in range(1, 9)],
        [(1, 10, 0)] + [(f, 99, 0) for f in range(2, 9)])
    one = target(evaluate_target_centric(gt, pred, lost_gap_frames=2,
                                          max_reacquisition_frames=3), 1)
    assert one["wrong_identity_frames"] == 7
    assert one["locked_id_wrong_person_frames"] == 0
    assert one["reacquisition_episodes"] == 1
    assert one["failed_reacquisitions"] == 1
    assert one["right_censored_episodes"] == 0


def test_unverified_prediction_does_not_count_as_wrong_person(tmp_path):
    gt, pred = write_case(tmp_path, [(1, 1, 0), (2, 1, 0)],
                          [(1, 10, 0), (2, 10, 70)])
    one = target(evaluate_target_centric(gt, pred), 1)
    assert one["locked_id_wrong_person_frames"] == 0
    assert one["locked_id_unmatched_frames"] == 1
    assert one["missed_frames"] == 1


def test_pooling_recomputes_counts_and_macro(tmp_path):
    gt, pred = write_case(tmp_path, [(1, 1, 0), (2, 1, 0)],
                          [(1, 10, 0), (2, 10, 0)])
    result = evaluate_target_centric(gt, pred)
    pooled = aggregate_target_centric([result, result])
    assert pooled["targets_evaluated"] == 2
    assert pooled["correct_identity_frames"] == 4
    assert pooled["target_identity_retention_rate"] == 1.0
    assert pooled["protocol_version"] == 2


def test_invalid_parameters_fail_fast(tmp_path):
    gt, pred = write_case(tmp_path, [(1, 1, 0)], [(1, 10, 0)])
    with pytest.raises(ValueError):
        evaluate_target_centric(gt, pred, lost_gap_frames=0)
    with pytest.raises(ValueError):
        evaluate_target_centric(gt, pred, iou_threshold=1.1)
