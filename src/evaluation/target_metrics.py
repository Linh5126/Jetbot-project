"""Target-centric evaluation for *offline* MOT17 identity logs.

This is a diagnostic protocol, not an official MOTChallenge score and not a
measurement of the physical robot controller. Each GT person is selected at
its first IoU-matched detection; that predicted persistent ID remains locked.

Crucially, two distinct errors are reported:
- target_id_mismatch_rate: the GT person was detected under a different ID.
- locked_id_wrong_person_rate: the *locked* ID was observed on another GT
  person, which is a closer proxy for the risk of following the wrong person.

Open loss episodes are right-censored unless the GT person remains observable
through the configured recovery deadline. A final out-of-view exit is not an
observed reacquisition failure.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


def _iou_xyxy(a, b) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / max(1e-12, area_a + area_b - inter)


def _read_gt(path: Path, min_visibility: float):
    by_frame, by_identity = defaultdict(list), defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split(",")
            if len(fields) < 7:
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                continue
            if not np.all(np.isfinite(values)):
                continue
            frame_id, gt_id = int(values[0]), int(values[1])
            x, y, width, height, confidence = values[2:7]
            category = int(values[7]) if len(values) > 7 else 1
            visibility = float(values[8]) if len(values) > 8 else 1.0
            if category != 1 or confidence <= 0 or visibility < min_visibility or width <= 0 or height <= 0:
                continue
            row = {"gt_id": gt_id, "bbox": [x, y, x + width, y + height]}
            by_frame[frame_id].append(row)
            by_identity[gt_id][frame_id] = row
    return by_frame, by_identity


def _read_identity_log(path: Path):
    by_frame = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "target_id", "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"identity_log.csv thiếu cột: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                frame_id, target_id = int(float(row["frame"])), int(float(row["target_id"]))
                box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
            except (TypeError, ValueError, KeyError):
                continue
            if target_id < 0 or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            by_frame[frame_id].append({"target_id": target_id, "bbox": box})
    return by_frame


def _match_indices(gt_rows: list[dict], pred_rows: list[dict], threshold: float) -> list[tuple[int, int]]:
    if not gt_rows or not pred_rows:
        return []
    ious = np.asarray([
        [_iou_xyxy(gt["bbox"], pred["bbox"]) for pred in pred_rows]
        for gt in gt_rows
    ], dtype=np.float64)
    # Require SciPy's optimal matching; a greedy fallback silently changes
    # evaluation semantics and would make benchmark results non-comparable.
    from scipy.optimize import linear_sum_assignment

    costs = np.where(ious >= threshold, 1.0 - ious, 1e6)
    rows, cols = linear_sum_assignment(costs)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if ious[r, c] >= threshold]


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator) / denominator if denominator else None


def _mean(values) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _median(values) -> Optional[float]:
    return float(np.median(values)) if values else None


def _recovery_episodes(
    first_frame: int,
    last_frame: int,
    visible: set[int],
    correct: set[int],
    wrong_follow: set[int],
    min_gap: int,
    deadline_frames: int,
) -> dict:
    """Count completed/observable episodes; separate unresolved censored ones."""
    outcomes = {
        "reacquisition_episodes": 0,
        "successful_reacquisitions": 0,
        "failed_reacquisitions": 0,
        "right_censored_episodes": 0,
        "late_reacquisitions": 0,
        "wrong_person_reacquisitions": 0,
        "recovery_latencies_frames": [],
    }
    run_start = None

    def close(run_end: int, recovery_frame: Optional[int]) -> None:
        if run_start is None or run_end - run_start + 1 < min_gap:
            return
        elapsed = (recovery_frame if recovery_frame is not None else last_frame) - (run_start - 1)
        wrong_during_gap = any(run_start <= frame <= run_end for frame in wrong_follow)
        eligible = False
        if recovery_frame is not None:
            eligible = True
            outcomes["reacquisition_episodes"] += 1
            if elapsed <= deadline_frames:
                outcomes["successful_reacquisitions"] += 1
                outcomes["recovery_latencies_frames"].append(elapsed)
            else:
                outcomes["failed_reacquisitions"] += 1
                outcomes["late_reacquisitions"] += 1
        else:
            deadline_frame = run_start - 1 + deadline_frames
            # An unresolved run is a failure only if the GT target can still
            # be observed at/after the deadline. Otherwise the observation is
            # truncated (out of view or end of sequence): right-censored.
            if elapsed >= deadline_frames and any(frame >= deadline_frame for frame in visible):
                eligible = True
                outcomes["reacquisition_episodes"] += 1
                outcomes["failed_reacquisitions"] += 1
            else:
                outcomes["right_censored_episodes"] += 1
        if wrong_during_gap and eligible:
            outcomes["wrong_person_reacquisitions"] += 1

    for frame in range(first_frame + 1, last_frame + 1):
        if frame in correct:
            if run_start is not None:
                close(frame - 1, frame)
                run_start = None
        elif run_start is None:
            run_start = frame
    if run_start is not None:
        close(last_frame, None)
    return outcomes


_SUM_KEYS = (
    "visible_frames", "matched_frames", "correct_identity_frames",
    "wrong_identity_frames", "missed_frames", "locked_id_verified_frames",
    "locked_id_correct_person_frames", "locked_id_wrong_person_frames",
    "locked_id_unmatched_frames", "reacquisition_episodes",
    "successful_reacquisitions", "failed_reacquisitions",
    "right_censored_episodes", "late_reacquisitions",
    "wrong_person_reacquisitions",
)


def _summarize(details: list[dict]) -> dict:
    totals = {key: sum(int(item[key]) for item in details) for key in _SUM_KEYS}
    latencies = [lat for item in details for lat in item["recovery_latencies_frames"]]
    return {
        **totals,
        "tracking_coverage": _safe_ratio(totals["matched_frames"], totals["visible_frames"]),
        "target_identity_retention_rate": _safe_ratio(totals["correct_identity_frames"], totals["visible_frames"]),
        "identity_accuracy_when_detected": _safe_ratio(totals["correct_identity_frames"], totals["matched_frames"]),
        "target_id_mismatch_rate": _safe_ratio(totals["wrong_identity_frames"], totals["matched_frames"]),
        # Backwards-compatible alias, NOT a measure of locked-ID following someone else.
        "wrong_person_assignment_rate": _safe_ratio(totals["wrong_identity_frames"], totals["matched_frames"]),
        "locked_id_wrong_person_rate": _safe_ratio(totals["locked_id_wrong_person_frames"], totals["locked_id_verified_frames"]),
        "reacquisition_success_rate": _safe_ratio(totals["successful_reacquisitions"], totals["reacquisition_episodes"]),
        "wrong_person_reacquisition_rate": _safe_ratio(totals["wrong_person_reacquisitions"], totals["reacquisition_episodes"]),
        "median_recovery_latency_frames": _median(latencies),
        "macro_target_tracking_coverage": _mean(item["tracking_coverage"] for item in details),
        "macro_target_identity_retention_rate": _mean(item["identity_retention_rate"] for item in details),
        "macro_target_identity_accuracy_when_detected": _mean(item["identity_accuracy_when_detected"] for item in details),
        "macro_target_wrong_person_assignment_rate": _mean(item["target_id_mismatch_rate"] for item in details),
        "macro_target_locked_id_wrong_person_rate": _mean(item["locked_id_wrong_person_rate"] for item in details),
        "macro_target_reacquisition_success_rate": _mean(item["reacquisition_success_rate"] for item in details),
    }


def evaluate_target_centric(
    gt_path,
    identity_log_path,
    iou_threshold: float = 0.5,
    min_visibility: float = 0.0,
    lost_gap_frames: int = 5,
    max_reacquisition_frames: int = 90,
    max_frame: Optional[int] = None,
    include_target_details: bool = True,
) -> dict:
    if not 0.0 < iou_threshold <= 1.0 or not 0.0 <= min_visibility <= 1.0:
        raise ValueError("IoU threshold phải thuộc (0,1], visibility thuộc [0,1]")
    if lost_gap_frames < 1 or max_reacquisition_frames < 1:
        raise ValueError("lost_gap_frames và max_reacquisition_frames phải >= 1")

    gt_by_frame, gt_by_id = _read_gt(Path(gt_path), min_visibility)
    pred_by_frame = _read_identity_log(Path(identity_log_path))
    if max_frame is not None:
        gt_by_frame = {f: rows for f, rows in gt_by_frame.items() if f <= max_frame}
        pred_by_frame = {f: rows for f, rows in pred_by_frame.items() if f <= max_frame}
        gt_by_id = {gid: {f: row for f, row in obs.items() if f <= max_frame} for gid, obs in gt_by_id.items()}
        gt_by_id = {gid: obs for gid, obs in gt_by_id.items() if obs}
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    if not frames:
        return {"warning": "No GT or prediction rows available for target-centric evaluation."}
    last_frame = int(max_frame) if max_frame is not None else frames[-1]

    # Store GT->persistent-ID matches AND owner of each predicted persistent ID
    # at each frame. Unmatched predicted IDs stay unknown, never labeled 'wrong person'.
    gt_assignment, id_owners, unmatched_ids = {}, {}, {}
    for frame in frames:
        gts, preds = gt_by_frame.get(frame, []), pred_by_frame.get(frame, [])
        pairs = _match_indices(gts, preds, iou_threshold)
        gt_assignment[frame] = {int(gts[r]["gt_id"]): int(preds[c]["target_id"]) for r, c in pairs}
        matched_columns = {c for _, c in pairs}
        owners = defaultdict(set)
        for r, c in pairs:
            owners[int(preds[c]["target_id"])].add(int(gts[r]["gt_id"]))
        id_owners[frame] = owners
        unmatched_ids[frame] = {int(preds[c]["target_id"]) for c in range(len(preds)) if c not in matched_columns}

    details = []
    for gt_id, observations in sorted(gt_by_id.items()):
        matched = [frame for frame in sorted(observations) if gt_id in gt_assignment.get(frame, {})]
        if not matched:
            continue  # Could not simulate initial target selection.
        initial = matched[0]
        locked_id = gt_assignment[initial][gt_id]
        visible = {frame for frame in observations if frame >= initial}
        correct = {frame for frame in visible if gt_assignment.get(frame, {}).get(gt_id) == locked_id}
        mismatched = {frame for frame in visible if gt_id in gt_assignment.get(frame, {}) and frame not in correct}
        wrong_follow, verified_correct, unknown = set(), set(), set()
        for frame in range(initial, last_frame + 1):
            owners = id_owners.get(frame, {}).get(locked_id, set())
            if any(owner != gt_id for owner in owners):
                wrong_follow.add(frame)
            elif gt_id in owners:
                verified_correct.add(frame)
            if locked_id in unmatched_ids.get(frame, set()):
                unknown.add(frame)
        loss = _recovery_episodes(initial, last_frame, visible, correct, wrong_follow,
                                  int(lost_gap_frames), int(max_reacquisition_frames))
        detail = {
            "gt_person_id": int(gt_id), "locked_target_id": int(locked_id),
            "initial_selection_frame": int(initial), "visible_frames": len(visible),
            "matched_frames": len(correct) + len(mismatched),
            "correct_identity_frames": len(correct), "wrong_identity_frames": len(mismatched),
            "missed_frames": len(visible) - len(correct) - len(mismatched),
            "locked_id_verified_frames": len(wrong_follow | verified_correct),
            "locked_id_correct_person_frames": len(verified_correct),
            "locked_id_wrong_person_frames": len(wrong_follow),
            "locked_id_unmatched_frames": len(unknown), **loss,
        }
        detail.update({
            "tracking_coverage": _safe_ratio(detail["matched_frames"], detail["visible_frames"]),
            "identity_retention_rate": _safe_ratio(detail["correct_identity_frames"], detail["visible_frames"]),
            "identity_accuracy_when_detected": _safe_ratio(detail["correct_identity_frames"], detail["matched_frames"]),
            "target_id_mismatch_rate": _safe_ratio(detail["wrong_identity_frames"], detail["matched_frames"]),
            "wrong_person_assignment_rate": _safe_ratio(detail["wrong_identity_frames"], detail["matched_frames"]),
            "locked_id_wrong_person_rate": _safe_ratio(detail["locked_id_wrong_person_frames"], detail["locked_id_verified_frames"]),
            "reacquisition_success_rate": _safe_ratio(detail["successful_reacquisitions"], detail["reacquisition_episodes"]),
        })
        details.append(detail)
    if not details:
        return {"protocol": "simulated-first-matched-selection", "targets_evaluated": 0,
                "warning": "No GT identities could be initialized from predictions."}
    result = {
        "protocol": "simulated-first-matched-selection; locked persistent ID; GT-IoU matching",
        "protocol_version": 2,
        "protocol_note": (
            "Diagnostic only; locked_id_wrong_person_rate requires GT matching; "
            "unmatched locked IDs remain unverified. Legacy wrong_person_assignment_rate "
            "is an alias for target_id_mismatch_rate. Right-censored episodes excluded "
            "from reacquisition_success_rate denominator."
        ),
        "iou_threshold": float(iou_threshold), "min_visibility": float(min_visibility),
        "lost_gap_frames": int(lost_gap_frames),
        "max_reacquisition_frames": int(max_reacquisition_frames),
        "targets_evaluated": len(details),
        **_summarize(details),
    }
    if include_target_details:
        result["target_details"] = details
    return result


def aggregate_target_centric(sequence_results: list[dict]) -> dict:
    valid = [item for item in sequence_results if item and "warning" not in item]
    if not valid:
        return {"warning": "No valid target-centric sequence results."}
    targets = [target for item in valid for target in item.get("target_details", [])]
    if not targets:
        raise ValueError("Aggregation requires include_target_details=True for all sequence results")
    return {
        "protocol_version": 2,
        "sequences_included": len(valid),
        "targets_evaluated": len(targets),
        **_summarize(targets),
        "protocol_note": "Pooled target-centric diagnostic; not an official MOTChallenge metric.",
    }
