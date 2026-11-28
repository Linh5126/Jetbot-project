"""On-board (Nano) pipeline parts that run without TensorRT: tracker, decode, selection, modes."""
import numpy as np
import pytest
import yaml

from onboard.bytetrack import ByteTracker
from onboard.models import letterbox, reid_batch, yolo_decode
from onboard.modes import NearestMode, nearest_to_centre
from onboard.selection import HandRaiseSelector, hand_lift, hands_raised

TRACKER_CFG = yaml.safe_load(open("configs/bytetrack_custom.yaml"))


def _scene(seed, frames=120):
    rng = np.random.RandomState(seed)
    people = [dict(x=rng.uniform(50, 400), y=rng.uniform(40, 120), vx=rng.uniform(-4, 4),
                   w=rng.uniform(40, 90), h=rng.uniform(120, 220)) for _ in range(4)]
    for _ in range(frames):
        rows = []
        for p in people:
            p["x"] += p["vx"]
            if p["x"] < 0 or p["x"] > 440:
                p["vx"] *= -1
            if rng.rand() < 0.08:
                continue
            box = np.array([p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]]) + rng.normal(0, 2, 4)
            rows.append(list(box) + [float(np.clip(rng.normal(0.6, 0.25), 0.11, 0.99)), 0.0])
        yield np.array(rows, np.float32).reshape(-1, 6)


def test_bytetrack_matches_ultralytics():
    trackers = pytest.importorskip("ultralytics.trackers.byte_tracker")
    from ultralytics.engine.results import Boxes
    from ultralytics.utils import IterableSimpleNamespace
    for seed in range(3):
        ref = trackers.BYTETracker(IterableSimpleNamespace(**TRACKER_CFG))
        mine = ByteTracker(**TRACKER_CFG)
        for data in _scene(seed):
            expected = ref.update(Boxes(data, (270, 480)), np.zeros((270, 480, 3), np.uint8))
            got = mine.update(data[:, :4], data[:, 4])
            assert sorted(int(r[4]) for r in expected) == sorted(d["track_id"] for d in got)
            for row in expected:
                match = next(d for d in got if d["track_id"] == int(row[4]))
                assert np.allclose(match["bbox"], row[:4], atol=0.2)


def test_bytetrack_keeps_id_through_short_miss():
    tracker = ByteTracker(**TRACKER_CFG)
    box = np.array([[100, 50, 160, 230]], np.float32)
    ids = [d["track_id"] for f in range(5) for d in tracker.update(box + f, np.array([0.9]))]
    assert tracker.update(np.zeros((0, 4)), np.zeros(0)) == []
    again = tracker.update(box + 6, np.array([0.9]))
    assert set(ids) == {1} and again[0]["track_id"] == 1


def test_bytetrack_carries_keypoints_of_the_matched_detection():
    tracker = ByteTracker(**TRACKER_CFG)
    box = np.array([[100, 50, 160, 230]], np.float32)
    for f in range(3):
        out = tracker.update(box + f, np.array([0.9]), np.full((1, 17, 3), float(f), np.float32))
    assert out[0]["kpts"][0, 0] == 2.0
    assert tracker.update(box + 3, np.array([0.9]))[0]["kpts"] is None  # no keypoints given


def test_letterbox_and_decode_back_to_frame_pixels():
    frame = np.zeros((270, 480, 3), np.uint8)
    image, scale, pad = letterbox(frame, (288, 480))
    assert image.shape == (288, 480, 3) and scale == 1.0 and pad == (0, 9)
    out = np.zeros((1, 84, 3), np.float32)
    out[0, :5, 0] = [240, 144, 60, 180, 0.9]   # person, centre of the letterboxed image
    out[0, :5, 1] = [242, 145, 60, 180, 0.5]   # duplicate -> removed by NMS
    out[0, 5, 2], out[0, 4, 2] = 0.9, 0.05      # another class only
    boxes, scores, kpts = yolo_decode(out, scale, pad, frame.shape, conf=0.10)
    assert len(boxes) == 1 and scores[0] == pytest.approx(0.9) and kpts.shape == (1, 0, 3)
    assert np.allclose(boxes[0], [210, 45, 270, 225])


def test_pose_decode_maps_keypoints_to_frame_pixels():
    out = np.zeros((1, 56, 2), np.float32)
    out[0, :5, 0] = [240, 144, 60, 180, 0.9]
    out[0, 5:, 0] = np.tile([250.0, 109.0, 0.8], 17)   # every keypoint at (250, 109) letterboxed
    out[0, 4, 1] = 0.05                                 # below conf
    boxes, scores, kpts = yolo_decode(out, 1.0, (0, 9), (270, 480, 3), conf=0.10)
    assert kpts.shape == (1, 17, 3)
    assert np.allclose(kpts[0], [250, 100, 0.8])        # pad_y 9 removed, conf untouched


def test_reid_batch_shape_and_empty_crop():
    batch = reid_batch([np.full((90, 40, 3), 127, np.uint8), np.zeros((0, 0, 3), np.uint8)])
    assert batch.shape == (2, 3, 256, 128) and batch.dtype == np.float32


def _pose(wrist_dy, conf=0.9):
    """COCO-17 keypoints: shoulders at y=100, wrists at y=100+wrist_dy (negative = raised)."""
    k = np.zeros((17, 3), np.float32)
    k[5], k[6] = [90, 100, conf], [130, 100, conf]
    k[9], k[10] = [85, 100 + wrist_dy, conf], [135, 100 + wrist_dy, conf]
    return k


def test_hands_raised_needs_both_wrists_clearly_above_shoulders():
    assert hands_raised(_pose(-40), 270)             # 40 px = 0.15 of the frame > margin 0.05
    assert not hands_raised(_pose(-10), 270)         # 0.04: inside the deadband
    assert not hands_raised(_pose(60), 270)          # hands down
    one = _pose(-40)
    one[10, 1] = 160
    assert not hands_raised(one, 270)                # one hand only
    assert not hands_raised(_pose(-40, conf=0.3), 270)  # keypoints not trusted
    assert not hands_raised(None, 270)


def test_hand_lift_reports_each_side_and_untrusted_keypoints():
    assert hand_lift(_pose(-27), 270) == pytest.approx([0.1, 0.1])
    low = _pose(54)
    low[10, 2] = 0.2                                 # right wrist not trusted
    left, right = hand_lift(low, 270)
    assert left == pytest.approx(-0.2) and right is None
    assert hand_lift(None, 270) == [None, None]


def test_pose_check_explains_why_nobody_is_selected():
    pytest.importorskip("cv2")
    from onboard.pose_check import describe
    box = np.array([[60, 40, 160, 200]], np.float32)
    people, why = describe(box, np.array([0.9]), _pose(-40)[None], 270, 0.05)
    assert why is None and people[0][2] and people[0][0] == pytest.approx(160 / 270.0)
    assert "not above" in describe(box, np.array([0.9]), _pose(60)[None], 270, 0.05)[1]
    assert "not seen" in describe(box, np.array([0.9]), _pose(-40, conf=0.2)[None], 270, 0.05)[1]
    assert describe(box, np.array([0.1]), _pose(-40)[None], 270, 0.05)[1] == "no person seen"  # low score
    two = describe(np.repeat(box, 2, 0), np.array([0.9, 0.8]), np.stack([_pose(-40)] * 2), 270, 0.05)[1]
    assert two == "2 people raising hands"


def test_hand_raise_selector_needs_one_raised_person_held():
    person = {"track_id": 4, "kpts": _pose(-40), "embedding": np.ones(3), "crop_quality": 0.8}
    seated = [{"track_id": 10 + i, "kpts": _pose(60)} for i in range(4)]
    sel = HandRaiseSelector(3, 270)
    assert sel.update([person] + seated) is None and sel.update(seated + [person]) is None
    assert sel.update([person]) is person and sel.raised == 1
    sel = HandRaiseSelector(2, 270)
    other = dict(person, track_id=5)
    assert sel.update([person, other]) is None and sel.update([person, other]) is None  # two raised
    assert "2 people" in sel.why
    poor = dict(person, crop_quality=0.1)
    assert HandRaiseSelector(1, 270).update([poor]) is None  # Re-ID reference would be weak


CFG = yaml.safe_load(open("configs/jetbot.yaml", encoding="utf-8"))


def test_nearest_to_centre_picks_the_box_closest_to_the_image_centre():
    boxes = np.array([[0, 27, 96, 243], [220, 30, 300, 250], [400, 0, 480, 270]], np.float32)
    assert nearest_to_centre(boxes, 480, 270) == 1
    assert nearest_to_centre(np.zeros((0, 4), np.float32), 480, 270) is None


def test_nearest_mode_settles_then_turns_toward_the_person_and_stops_without_one():
    mode = NearestMode(CFG, 270)
    frame = np.zeros((270, 480, 3), np.uint8)
    box = np.array([[330, 90, 390, 240]], np.float32)   # right of centre, h 0.56 -> approach
    kpts = np.zeros((1, 17, 3), np.float32)
    for f in range(CFG["onboard"]["follow"]["settle_frames"]):
        command, fields = mode.step(frame, f, box, np.array([0.9]), kpts, 0.0)
    assert fields["reason"] == "APPROACH" and command.forward > 0 and command.turn < 0  # negative = right
    command, fields = mode.step(frame, 9, np.zeros((0, 4), np.float32), np.zeros(0), kpts[:0], 0.0)
    assert command.forward == 0 and command.turn == 0 and fields["state"] == "NO_PERSON"
