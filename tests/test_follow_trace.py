"""Synthetic MOT17 GT/PC follow logs: no GPU or footage needed."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.evaluation.follow_trace import FollowTraceWriter, GroundTruthIndex, evaluate_follow_trace
from src.following.controller import MotionCommand
from src.following.guarded_lock import GuardConfig, GuardedTargetLock
from src.following.target_lock import FollowState, LockResult, TargetObservation


def fake_gt(tmp_path: Path) -> GroundTruthIndex:
    sequence = tmp_path / "MOT17-11-FRCNN"
    (sequence / "gt").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-11-FRCNN\nimWidth=640\nimHeight=480\n", encoding="utf-8")
    lines = []
    for frame in range(1, 8):
        lines.append(f"{frame},12,100,100,100,200,1,1,1\n")
        lines.append(f"{frame},99,340,100,100,200,1,1,1\n")
    (sequence / "gt" / "gt.txt").write_text("".join(lines), encoding="utf-8")
    return GroundTruthIndex(sequence)


def lock_result(state: FollowState, bbox=None) -> LockResult:
    observation = (TargetObservation(7, 3, bbox, 0.95) if bbox is not None else None)
    return LockResult(state, 7, observation, "unit-test")


def test_gt_attribution_and_video_size_guard(tmp_path):
    index = fake_gt(tmp_path)
    index.validate_frame_size(640, 480)
    with pytest.raises(ValueError, match="aligned"):
        index.validate_frame_size(320, 240)
    assert index.match_box(3, [100, 100, 200, 300]) == (12, "MATCHED")
    assert index.match_box(3, [340, 100, 440, 300]) == (99, "MATCHED")
    assert index.match_box(3, [50, 10, 70, 30]) == (None, "UNMATCHED")


def test_trace_reports_wrong_person_authorization_and_hold_without_misnaming(tmp_path):
    index = fake_gt(tmp_path)
    trace_path = tmp_path / "following.csv"
    target = (100, 100, 200, 300)
    impostor = (340, 100, 440, 300)
    with FollowTraceWriter(trace_path) as writer:
        writer.write(1, 30, 1, 12, lock_result(FollowState.TRACKING, target),
                     MotionCommand(forward=0.1), event="SELECT")
        writer.write(2, 30, 1, 12, lock_result(FollowState.LOST), MotionCommand())
        writer.write(3, 30, 1, 12, lock_result(FollowState.VERIFYING), MotionCommand())
        writer.write(4, 30, 1, 12, lock_result(FollowState.REACQUIRED, impostor),
                     MotionCommand(forward=0.1))
        writer.write(5, 30, 1, 12, lock_result(FollowState.LOST), MotionCommand())
        writer.write(6, 30, 1, 12, lock_result(FollowState.REACQUIRED, target),
                     MotionCommand(forward=0.1))
    summary = evaluate_follow_trace(trace_path, index)
    assert summary["sessions_with_gt"] == 1
    assert summary["evaluated_frames"] == 6
    assert summary["gt_visible_frames"] == 6
    assert summary["authorized_frames"] == 3
    assert summary["correct_authorizations"] == 2
    assert summary["wrong_person_authorizations"] == 1
    assert summary["wrong_person_nonzero_commands"] == 1
    assert summary["hold_while_gt_visible_frames"] == 3
    assert summary["wrong_person_recoveries"] == 1
    assert summary["correct_recoveries"] == 1
    assert summary["authorized_wrong_person_rate_known_gt"] == pytest.approx(1/3)
    assert summary["recovery_latency_frames_mean"] == 1
    assert summary["correct_authorization_per_visible_gt_frame"] == pytest.approx(2/6)
    assert list(csv.DictReader(trace_path.open(encoding="utf-8")))[0]["event"] == "SELECT"


def test_guarded_demo_never_writes_command_while_denied(tmp_path):
    index = fake_gt(tmp_path)
    path = tmp_path / "trace.csv"
    with FollowTraceWriter(path) as writer:
        writer.write(1, 30, 1, 12, lock_result(FollowState.LOST),
                     MotionCommand(forward=0.1))
    with pytest.raises(ValueError, match="Unauthorized|unauthorized"):
        evaluate_follow_trace(path, index)


def test_missing_gt_selection_is_not_scored(tmp_path):
    gt = fake_gt(tmp_path)
    path = tmp_path / "missing.csv"
    with FollowTraceWriter(path) as writer:
        writer.write(1, 30, 1, None, lock_result(FollowState.TRACKING,
                     (100, 100, 200, 300)), MotionCommand())
    metrics = evaluate_follow_trace(path, gt)
    assert metrics["sessions_with_gt"] == 0
    assert metrics["warning"] is not None


def test_gt_overlap_ties_are_ambiguous_not_wrong(tmp_path):
    index = fake_gt(tmp_path)
    index.by_frame[1].append((123, (100, 100, 200, 300)))
    assert index.match_box(1, (100, 100, 200, 300)) == (None, "AMBIGUOUS")
