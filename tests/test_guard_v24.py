"""V2.4 identity continuity / actuation safety regression tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.following.controller import FollowController, MotionCommand
from src.following.guarded_lock import GuardConfig, GuardedTargetLock
from src.following.target_lock import FollowState, TargetObservation

REF = np.array([1.0, 0.0, 0.0], dtype=np.float32)
WRONG = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def person(track=3, emb=REF, box=(100, 50, 200, 350)):
    return {"target_id": track + 10, "track_id": track, "confidence": 0.9,
            "crop_quality": 0.9, "bbox": box, "embedding": emb}


def selected(**kwargs):
    manager = GuardedTargetLock(GuardConfig(
        confirm_frames=5, same_track_confirm_frames=2, candidate_grace_frames=2,
        min_final_clear_frames=2, **kwargs))
    d = person()
    manager.select(TargetObservation.from_detection(d), 1, [d])
    return manager


def assert_held(result):
    assert not result.may_follow
    assert FollowController().command(result, 640, 480) == MotionCommand()


def test_identity_survives_occlusion_while_motion_stops():
    lock = selected()
    first = lock.update(2, [person()])
    assert first.may_follow
    hold = lock.update(3, [])
    assert hold.state == FollowState.LOST
    assert_held(hold)
    assert lock.diagnostics.identity_locked
    assert not lock.diagnostics.movement_authorized
    assert lock.target_id == 13
    verifying = lock.update(4, [person()])
    assert verifying.state == FollowState.VERIFYING
    assert_held(verifying)
    recovered = lock.update(5, [person()])
    assert recovered.state == FollowState.REACQUIRED and recovered.may_follow
    assert lock.target_id == 13
    assert lock.diagnostics.trusted_track_id == 3


def test_new_id_uses_five_good_votes_with_grace_but_never_moves_in_gap():
    lock = selected()
    lock.update(2, [])
    assert lock.update(3, [person(track=8)]).state == FollowState.VERIFYING
    assert lock.diagnostics.candidate_hits == 1
    assert_held(lock.update(4, []))
    assert lock.diagnostics.candidate_hits == 1
    assert_held(lock.update(5, []))
    assert lock.diagnostics.candidate_hits == 1
    # After two-frame gap, last two clear samples are still necessary.
    for frame in (6, 7, 8):
        assert_held(lock.update(frame, [person(track=8)]))
    assert lock.diagnostics.candidate_hits == 4
    assert lock.update(9, [person(track=8)]).state == FollowState.REACQUIRED
    assert lock.target_id == 13


def test_gap_beyond_grace_clears_progress():
    lock = selected()
    lock.update(2, [])
    lock.update(3, [person(track=8)])
    for frame in (4, 5, 6):
        assert_held(lock.update(frame, []))
    assert_held(lock.update(7, [person(track=8)]))
    assert lock.diagnostics.candidate_hits == 1


def test_ambiguous_evidence_clears_progress_and_cannot_reuse_votes():
    lock = selected()
    lock.update(2, [])
    lock.update(3, [person(track=8)])
    same_clothes = np.array([0.999, 0.045, 0], dtype=np.float32)
    hold = lock.update(4, [person(track=8), person(track=9, emb=same_clothes,
                                               box=(380, 50, 480, 350))])
    assert_held(hold)
    assert lock.diagnostics.candidate_hits == 0
    assert lock.diagnostics.reason_code == "AMBIGUOUS"
    for frame in (5, 6, 7, 8):
        assert_held(lock.update(frame, [person(track=8)]))
    assert lock.update(9, [person(track=8)]).may_follow


def test_overlap_freezes_candidate_but_not_permission():
    lock = selected()
    lock.update(2, [])
    lock.update(3, [person(track=8)])
    occluder = person(track=15, emb=WRONG, box=(115, 50, 215, 350))
    assert_held(lock.update(4, [person(track=8), occluder]))
    assert lock.diagnostics.reason_code == "OCCLUDED"
    assert lock.diagnostics.candidate_hits == 1
    for frame in (5, 6, 7):
        assert_held(lock.update(frame, [person(track=8)]))
    assert lock.update(8, [person(track=8)]).may_follow


def test_tracker_hijack_and_different_person_never_authorized():
    lock = selected()
    assert_held(lock.update(2, [person(track=3, emb=WRONG)]))
    assert_held(lock.update(3, [person(track=3, emb=WRONG)]))
    assert lock.target_id == 13 and lock.diagnostics.identity_locked
    assert_held(lock.update(4, [person(track=31, emb=WRONG)]))
    assert lock.diagnostics.trusted_track_id == 3


def test_duplicate_tracker_id_and_timeout_fail_closed():
    lock = selected(max_lost_frames=5)
    assert_held(lock.update(2, [person(track=3), person(track=3)]))
    assert lock.diagnostics.reason_code == "DUPLICATE_TRACK"
    for frame in (3, 4, 5, 6):
        assert_held(lock.update(frame, []))
    result = lock.update(7, [person()])
    assert result.state == FollowState.STOPPED
    assert_held(result)


def test_guard_configuration_validation():
    with pytest.raises(ValueError):
        GuardConfig(candidate_grace_frames=-1)
    with pytest.raises(ValueError):
        GuardConfig(same_track_recovery_distance=0.8)
    with pytest.raises(ValueError):
        GuardConfig(confirm_frames=1, min_final_clear_frames=2)
