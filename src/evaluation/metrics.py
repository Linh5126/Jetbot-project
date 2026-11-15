
from __future__ import annotations

import numpy as np
from src.data.mot17 import read_mot_gt, read_mot_result

# Compatibility with motmetrics on NumPy 2.x
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


def to_xywh(rows):
    """
    Convert internal XYXY boxes to the XYWH format
    required by motmetrics.iou_matrix().
    """
    boxes = []

    for row in rows:
        x1, y1, x2, y2 = row[2]
        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

    return np.asarray(boxes, dtype=np.float64).reshape((-1, 4))


def _metric_for_ids(gt_rows, pred_rows):
    try:
        import motmetrics as mm
    except ImportError:
        return {
            "warning": "Missing motmetrics. Run: python -m pip install motmetrics"
        }

    accumulator = mm.MOTAccumulator(auto_id=False)

    gt_by_frame = {}
    pred_by_frame = {}

    for row in gt_rows:
        gt_by_frame.setdefault(row[0], []).append(row)

    for row in pred_rows:
        pred_by_frame.setdefault(row[0], []).append(row)

    frames = sorted(
        set(gt_by_frame.keys()) | set(pred_by_frame.keys())
    )

    for frame_id in frames:
        gt = gt_by_frame.get(frame_id, [])
        pred = pred_by_frame.get(frame_id, [])

        gt_ids = [r[1] for r in gt]
        pred_ids = [r[1] for r in pred]

        gt_boxes = to_xywh(gt)
        pred_boxes = to_xywh(pred)

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            distances = mm.distances.iou_matrix(
                gt_boxes,
                pred_boxes,
                max_iou=0.5
            )
        else:
            distances = np.empty(
                (len(gt_boxes), len(pred_boxes)),
                dtype=np.float64
            )

        # Pass actual frame ID because auto_id=False.
        accumulator.update(
            gt_ids,
            pred_ids,
            distances,
            frameid=int(frame_id)
        )

    metrics = [
        "idf1",
        "mota",
        "motp",
        "num_switches",
        "num_false_positives",
        "num_misses",
    ]

    summary = mm.metrics.create().compute(
        accumulator,
        metrics=metrics,
        name="sequence"
    )

    values = summary.loc["sequence"].to_dict()

    result = {}

    for key, value in values.items():
        try:
            result[key] = (
                None if not np.isfinite(value) else float(value)
            )
        except (TypeError, ValueError):
            result[key] = None

    return result


def evaluate_mot(
    gt_path,
    target_result_path,
    raw_result_path,
    max_frame=None
):
    gt_rows = read_mot_gt(gt_path)
    target_rows = read_mot_result(target_result_path)
    raw_rows = read_mot_result(raw_result_path)

    # Restrict GT and predictions to processed frames.
    if max_frame is not None:
        max_frame = int(max_frame)

        gt_rows = [
            r for r in gt_rows if r[0] <= max_frame
        ]

        target_rows = [
            r for r in target_rows if r[0] <= max_frame
        ]

        raw_rows = [
            r for r in raw_rows if r[0] <= max_frame
        ]

    if not gt_rows:
        return {
            "warning": f"No valid GT rows found: {gt_path}"
        }

    return {
        "identity_manager_ids": _metric_for_ids(
            gt_rows, target_rows
        ),

        "bytetrack_raw_ids": _metric_for_ids(
            gt_rows, raw_rows
        ),

        "protocol_note": (
            "Diagnostic evaluation on MOT17 train GT. "
            "Not an official MOTChallenge score."
        )
    }


def evaluate_mot_collection(sequence_records):
    """Compute pooled MOT metrics across sequences with namespaced identities.

    sequence_records is an iterable of dictionaries containing:
      sequence, gt_path, target_result_path, raw_result_path, max_frame(optional).

    Frame IDs are offset per sequence and object IDs are namespaced per sequence,
    preventing ID collisions between different MOT17 cameras.
    """
    try:
        import motmetrics as mm
    except ImportError:
        return {"warning": "Missing motmetrics. Run: python -m pip install motmetrics"}

    target_acc = mm.MOTAccumulator(auto_id=False)
    raw_acc = mm.MOTAccumulator(auto_id=False)
    frame_offset = 0
    sequences_used = 0

    # motmetrics stores object/hypothesis IDs in its event DataFrame using
    # numeric columns (OId/HId). String namespaces such as
    # "MOT17-02-FRCNN:2" therefore fail later when pandas casts OId/HId to
    # float. Keep the per-sequence namespace, but encode each namespace as a
    # stable positive integer. Use separate registries for GT, target and raw
    # tracker IDs because these are independent identity spaces.
    gt_id_registry = {}
    target_id_registry = {}
    raw_id_registry = {}

    def numeric_namespaced_id(registry, sequence_name, local_id):
        key = (str(sequence_name), int(local_id))
        if key not in registry:
            registry[key] = len(registry) + 1
        return registry[key]

    for record in sequence_records:
        sequence = str(record["sequence"])
        gt_rows = read_mot_gt(record["gt_path"])
        target_rows = read_mot_result(record["target_result_path"])
        raw_rows = read_mot_result(record["raw_result_path"])
        max_frame = record.get("max_frame")
        if max_frame is not None:
            max_frame = int(max_frame)
            gt_rows = [r for r in gt_rows if r[0] <= max_frame]
            target_rows = [r for r in target_rows if r[0] <= max_frame]
            raw_rows = [r for r in raw_rows if r[0] <= max_frame]

        gt_by_frame, target_by_frame, raw_by_frame = {}, {}, {}
        for row in gt_rows:
            gt_by_frame.setdefault(row[0], []).append(row)
        for row in target_rows:
            target_by_frame.setdefault(row[0], []).append(row)
        for row in raw_rows:
            raw_by_frame.setdefault(row[0], []).append(row)

        frames = sorted(set(gt_by_frame) | set(target_by_frame) | set(raw_by_frame))
        if not frames:
            continue
        local_max = max(frames)
        for frame_id in frames:
            gt = gt_by_frame.get(frame_id, [])
            pred_target = target_by_frame.get(frame_id, [])
            pred_raw = raw_by_frame.get(frame_id, [])
            gt_boxes = to_xywh(gt)
            target_boxes = to_xywh(pred_target)
            raw_boxes = to_xywh(pred_raw)
            if len(gt_boxes) and len(target_boxes):
                target_dist = mm.distances.iou_matrix(gt_boxes, target_boxes, max_iou=0.5)
            else:
                target_dist = np.empty((len(gt_boxes), len(target_boxes)), dtype=np.float64)
            if len(gt_boxes) and len(raw_boxes):
                raw_dist = mm.distances.iou_matrix(gt_boxes, raw_boxes, max_iou=0.5)
            else:
                raw_dist = np.empty((len(gt_boxes), len(raw_boxes)), dtype=np.float64)

            namespaced_gt = [
                numeric_namespaced_id(gt_id_registry, sequence, r[1])
                for r in gt
            ]
            namespaced_target = [
                numeric_namespaced_id(target_id_registry, sequence, r[1])
                for r in pred_target
            ]
            namespaced_raw = [
                numeric_namespaced_id(raw_id_registry, sequence, r[1])
                for r in pred_raw
            ]
            synthetic_frame = frame_offset + int(frame_id)
            target_acc.update(namespaced_gt, namespaced_target, target_dist, frameid=synthetic_frame)
            raw_acc.update(namespaced_gt, namespaced_raw, raw_dist, frameid=synthetic_frame)

        frame_offset += int(local_max) + 1
        sequences_used += 1

    metrics = [
        "idf1", "mota", "motp", "num_switches",
        "num_false_positives", "num_misses"
    ]

    def compute(acc):
        summary = mm.metrics.create().compute(acc, metrics=metrics, name="pooled")
        values = summary.loc["pooled"].to_dict()
        result = {}
        for key, value in values.items():
            try:
                result[key] = None if not np.isfinite(value) else float(value)
            except (TypeError, ValueError):
                result[key] = None
        return result

    return {
        "identity_manager_ids": compute(target_acc),
        "bytetrack_raw_ids": compute(raw_acc),
        "sequences_included": sequences_used,
        "protocol_note": (
            "Pooled diagnostic MOT metrics on MOT17 train GT; identities are "
            "namespaced per sequence using numeric IDs for motmetrics compatibility. "
            "Not an official MOTChallenge score."
        ),
    }
