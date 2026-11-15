"""Regression tests for fail-closed target selection in crowded videos.

Pure Python/numpy: no GPU, camera, model weights, or JetBot required.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.following.controller import FollowController, MotionCommand
from src.following.guarded_lock import GuardConfig, GuardedTargetLock
from src.following.target_lock import FollowState, TargetObservation
from src.runtime.overlay import plan_box_labels

A = np.array([1., 0., 0.], np.float32)
B = np.array([0., 1., 0.], np.float32)
SIMILAR = np.array([0.998, 0.060, 0.], np.float32)


def det(track=3, target=7, emb=A, box=(100, 100, 200, 350), quality=0.9):
    return {"track_id": track, "target_id": target, "confidence": 0.91,
            "crop_quality": quality, "bbox": box, "embedding": emb}


def lock_target(manager, detection=None):
    detection = detection or det()
    return manager.select(TargetObservation.from_detection(detection), 1, [detection])


def guarded(confirm=3, max_lost=50):
    return GuardedTargetLock(GuardConfig(confirm_frames=confirm, max_lost_frames=max_lost))


def test_same_track_id_different_person_is_stopped_immediately():
    guard = guarded()
    assert lock_target(guard).may_follow
    result = guard.update(2, [det(emb=B)])  # ByteTrack B3 hijacked.
    assert result.state == FollowState.LOST and not result.may_follow
    assert FollowController().command(result, 640, 480) == MotionCommand()
    assert guard.target_id == 7  # No automatic identity rewrite.


def test_return_with_different_persistent_id_is_recovered_only_after_confirmation():
    guard = guarded(confirm=3)
    lock_target(guard)
    assert guard.update(2, [det(track=3, target=7, emb=B)]).state == FollowState.LOST
    for frame in (3, 4):
        result = guard.update(frame, [det(track=88, target=999, emb=A)])
        assert result.state == FollowState.VERIFYING and not result.may_follow
    recovered = guard.update(5, [det(track=88, target=999, emb=A)])
    assert recovered.state == FollowState.REACQUIRED and recovered.may_follow
    assert recovered.target_id == 7
    assert recovered.observation.track_id == 88
    assert guard.update(6, [det(track=88, target=999, emb=A)]).state == FollowState.TRACKING
    # UI must label the observed track, not a possibly hijacked V5 target_id.
    labels = plan_box_labels([det(track=88, target=999), det(track=3, target=7, emb=B,
                             box=(400, 100, 500, 350))], 7, 640, 480, active_track_id=88)
    assert sum(label.target for label in labels) == 1
    assert any(label.target and "T7 B88" in label.text for label in labels)


def test_ambiguous_twins_never_authorized_to_follow():
    guard = guarded(confirm=2)
    lock_target(guard)
    results = [guard.update(i, [
        det(track=44, target=7, emb=A),
        det(track=45, target=9, emb=SIMILAR, box=(400, 100, 500, 350)),
    ]) for i in range(2, 9)]
    assert all(not r.may_follow for r in results)
    assert all(r.state == FollowState.LOST for r in results)


def test_ambiguous_overlap_stops_even_with_perfect_embedding():
    guard = guarded()
    lock_target(guard)
    # Other person's bbox covers a large part of target bbox.
    result = guard.update(2, [
        det(), det(track=4, target=8, emb=B, box=(130, 100, 230, 350)),
    ])
    assert result.state == FollowState.LOST and not result.may_follow


def test_malformed_or_missing_embedding_fails_closed():
    guard = guarded()
    with pytest.raises(ValueError):
        lock_target(guard, det(emb=None))
    lock_target(guard)
    for frame in (2, 3):
        assert not guard.update(frame, [det(emb=None)]).may_follow
    assert guard.state == FollowState.LOST


def test_do_not_lock_onto_heavily_occluded_person():
    guard = guarded()
    person = det()
    occluder = det(track=4, target=8, emb=B, box=(160, 100, 240, 350))
    with pytest.raises(ValueError, match="overlaps"):
        guard.select(TargetObservation.from_detection(person), 1, [person, occluder])
    assert guard.state == FollowState.IDLE


def test_no_gallery_poisoning_from_wrong_id_and_recovery_to_original():
    guard = guarded(confirm=2)
    lock_target(guard)
    assert not guard.update(2, [det(emb=B)]).may_follow
    assert not guard.update(3, [det(track=8, target=70, emb=B)]).may_follow
    assert guard._distance(det(emb=A)) == pytest.approx(0.0)
    assert guard._distance(det(emb=B)) == pytest.approx(1.0)
    assert guard.update(4, [det(track=9, target=90, emb=A)]).state == FollowState.VERIFYING
    assert guard.update(5, [det(track=9, target=90, emb=A)]).may_follow


def test_timeout_and_motion_gate():
    guard = guarded(confirm=2, max_lost=4)
    lock_target(guard)
    assert not guard.update(2, [det(track=99, target=50, emb=A,
                                   box=(1200, 100, 1300, 350))]).may_follow
    for frame in (3, 4, 5):
        guard.update(frame, [])
    assert guard.update(6, [det()]).state == FollowState.STOPPED
    assert not guard.update(7, [det()]).may_follow


def test_candidate_track_change_resets_confirmation():
    guard = guarded(confirm=2)
    lock_target(guard)
    guard.update(2, [])
    assert guard.update(3, [det(track=4, target=99)]).state == FollowState.VERIFYING
    assert guard.update(4, [det(track=5, target=100)]).state == FollowState.VERIFYING
    assert guard.update(5, [det(track=5, target=100)]).may_follow


def test_config_validation():
    with pytest.raises(ValueError):
        GuardConfig(recovery_distance=0.9, track_distance=0.2)
    with pytest.raises(ValueError):
        GuardConfig(max_overlap=1.5)
