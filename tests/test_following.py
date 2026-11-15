from __future__ import annotations

import pytest

from src.following.controller import FollowController, MotionCommand
from src.following.driver import MockRobotDriver
from src.following.target_lock import FollowState, TargetLockManager, TargetObservation
from demo_pc import select_at_point


def det(tid=7, track=3, box=(250, 100, 390, 380), confidence=0.9):
    return {"target_id": tid, "track_id": track, "bbox": box, "confidence": confidence}


def selected(manager, frame=1):
    return manager.select(TargetObservation.from_detection(det()), frame)


def test_target_never_switches_automatically_and_requires_recovery_confirmation():
    manager = TargetLockManager(recovery_confirm_frames=3, max_lost_frames=20)
    assert selected(manager).may_follow
    assert manager.update(2, [det(9, 4)]).state == FollowState.LOST
    for i in (3, 4):
        r = manager.update(i, [det(track=100)])
        assert r.state == FollowState.VERIFYING and not r.may_follow
    r = manager.update(5, [det(track=100)])
    assert r.state == FollowState.REACQUIRED and r.may_follow
    assert manager.target_id == 7
    assert manager.update(6, [det(track=100)]).state == FollowState.TRACKING


def test_candidate_switch_resets_confirmation():
    manager = TargetLockManager(recovery_confirm_frames=2)
    selected(manager)
    manager.update(2, [])
    assert manager.update(3, [det(track=9)]).state == FollowState.VERIFYING
    assert manager.update(4, [det(track=10)]).state == FollowState.VERIFYING
    assert manager.update(5, [det(track=10)]).may_follow


def test_duplicate_target_id_fails_closed():
    manager = TargetLockManager()
    selected(manager)
    r = manager.update(2, [det(track=3), det(track=4)])
    assert r.state == FollowState.LOST and not r.may_follow


def test_timeout_blocks_follow_even_if_old_target_returns():
    manager = TargetLockManager(max_lost_frames=3)
    selected(manager)
    manager.update(2, [])
    manager.update(3, [])
    manager.update(4, [])
    assert manager.update(5, [det(track=50)]).state == FollowState.STOPPED
    assert not manager.update(6, [det(track=50)]).may_follow
    assert manager.target_id == 7


def test_manual_stop_is_terminal_until_new_selection():
    manager = TargetLockManager()
    selected(manager)
    assert not manager.stop().may_follow
    assert not manager.update(2, [det()]).may_follow
    manager.reset()
    assert manager.state == FollowState.IDLE
    assert selected(manager, frame=3).may_follow


def test_controller_stops_on_loss_and_unsafe_geometry():
    manager = TargetLockManager()
    selected(manager)
    controller = FollowController()
    assert controller.command(manager.update(2, []), 640, 480) == MotionCommand()
    active = manager.select(TargetObservation.from_detection(det()), 2)
    assert controller.command(active, 0, 480) == MotionCommand()
    assert controller.command(active, 640, 20) == MotionCommand()


def test_controller_turn_direction_and_forward_bounds():
    manager = TargetLockManager()
    controller = FollowController()
    right = manager.select(TargetObservation.from_detection(
        det(box=(420, 50, 540, 210))), 1)
    cmd = controller.command(right, 640, 480)
    assert 0 < cmd.forward <= controller.config.max_forward
    assert -controller.config.max_turn <= cmd.turn < 0
    left = manager.select(TargetObservation.from_detection(
        det(box=(60, 10, 160, 460))), 2)
    cmd = controller.command(left, 640, 480)
    assert cmd.turn > 0
    assert cmd.forward == 0  # No reversing when too close


def test_driver_and_selection_hit_testing():
    driver = MockRobotDriver()
    driver.send(MotionCommand(forward=0.12, turn=0.1))
    assert driver.last_command.forward == pytest.approx(0.12)
    driver.close()
    assert driver.last_command == MotionCommand()
    boxes = [det(tid=1, box=(10, 10, 100, 100)), det(tid=2, box=(20, 20, 50, 50))]
    assert select_at_point(boxes, 25, 25)["target_id"] == 2
    assert select_at_point(boxes, 200, 200) is None


def test_switching_bytetrack_id_without_missing_frame_requires_verification():
    manager = TargetLockManager(recovery_confirm_frames=2)
    selected(manager)
    assert manager.update(2, [det(track=33)]).state == FollowState.VERIFYING
    assert not manager.update(2 + 1, [det(track=44)]).may_follow
    assert manager.update(4, [det(track=44)]).state == FollowState.REACQUIRED
