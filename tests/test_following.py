from __future__ import annotations

import pytest

from src.following.controller import (FollowController, MotionCommand, StagedControllerConfig,
                                      StagedFollowController)
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


def staged(box, frames, ctrl=None, size=(480, 270)):
    ctrl = ctrl or StagedFollowController()
    for _ in range(frames):
        command = ctrl.command_for_box(box, *size)
    return ctrl, command


def test_staged_controller_settles_before_moving_and_resets_on_jump_or_loss():
    ctrl, command = staged((220, 60, 260, 210), 2)       # centred, h 0.56
    assert ctrl.phase == "SETTLING" and command == MotionCommand()
    ctrl, command = staged((220, 60, 260, 210), 1, ctrl)  # h 0.56: inside slow_band, slowing down
    assert ctrl.phase == "APPROACH" and command.forward == pytest.approx(0.12 * (0.70 - 150 / 270) / 0.15)
    assert command.turn == 0.0
    ctrl, command = staged((0, 60, 40, 210), 1, ctrl)     # no overlap with the last box
    assert ctrl.phase == "SETTLING" and command == MotionCommand()
    ctrl, command = staged(None, 1, ctrl)
    assert ctrl.phase == "IDLE" and command == MotionCommand()
    assert staged((220, 60, 260, 70), 5)[1] == MotionCommand()   # too small
    assert staged((260, 60, 220, 210), 5)[1] == MotionCommand()  # x2 < x1


def test_staged_controller_holds_close_with_hysteresis():
    ctrl, command = staged((200, 40, 280, 240), 3)       # h 0.74 >= 0.70
    assert ctrl.phase == "HOLD" and command.forward == 0
    ctrl, command = staged((200, 50, 280, 236), 1, ctrl)  # h 0.69: inside the margin, still HOLD
    assert ctrl.phase == "HOLD"
    ctrl, command = staged((200, 70, 280, 236), 1, ctrl)  # h 0.61 < 0.65: approach again, slowly
    assert ctrl.phase == "APPROACH" and command.forward == pytest.approx(0.12 * (0.70 - 166 / 270) / 0.15)
    ctrl, command = staged((200, 150, 280, 236), 3, StagedFollowController())  # h 0.32: far, full speed
    assert command.forward == pytest.approx(0.12)


def test_staged_controller_turn_sign_deadband_smoothing_and_turn_in_place():
    ctrl, command = staged((300, 60, 420, 210), 3)       # right of centre: side error 0.5
    assert command.turn == pytest.approx(0.3 * -0.25) and command.forward > 0  # one smoothed step
    ctrl, command = staged((300, 60, 420, 210), 30, ctrl)
    assert command.turn == pytest.approx(-0.25, abs=1e-3)  # negative = right
    ctrl, command = staged((220, 60, 330, 210), 1, ctrl)  # side error 0.146 <= deadband 0.15
    assert ctrl.phase == "APPROACH" and command.turn == 0.0
    ctrl, command = staged((440, 60, 480, 210), 3)       # side error 0.92 > max_side_error
    assert ctrl.phase == "HOLD" and command.forward == 0 and command.turn < 0  # turn in place
    with pytest.raises(ValueError):
        StagedControllerConfig(resume_margin=0.8)


def test_staged_controller_does_not_approach_when_the_head_is_cut_off():
    ctrl, command = staged((200, 0, 280, 160), 3)        # h 0.59 < stop, but the top touches the frame
    assert ctrl.phase == "HOLD" and command.forward == 0
    ctrl, command = staged((200, 5, 280, 160), 1, ctrl)   # top 0.019: barely back, keep holding
    assert ctrl.phase == "HOLD"
    ctrl, command = staged((200, 12, 280, 160), 1, ctrl)  # head clearly in view, h 0.55 < 0.65: approach
    assert ctrl.phase == "APPROACH"



def test_staged_controller_coasts_straight_through_a_short_flicker_only_when_far():
    ctrl = StagedFollowController(StagedControllerConfig(coast_frames=2))
    ctrl, command = staged((200, 150, 280, 236), 3, ctrl)  # h 0.32: far, full speed
    ctrl, command = staged(None, 2, ctrl)
    assert ctrl.phase == "COAST" and command == MotionCommand(forward=0.12)
    ctrl, command = staged((200, 150, 280, 236), 1, ctrl)  # back: no settling again
    assert ctrl.phase == "APPROACH"
    ctrl, command = staged(None, 3, ctrl)                  # gone too long
    assert ctrl.phase == "IDLE" and command == MotionCommand()
    ctrl = StagedFollowController(StagedControllerConfig(coast_frames=2))
    ctrl, command = staged((220, 60, 260, 210), 3, ctrl)   # h 0.56: slowing near the person
    ctrl, command = staged(None, 1, ctrl)
    assert ctrl.phase == "PAUSE" and command == MotionCommand()  # close: stop at once
    ctrl, command = staged((220, 60, 260, 210), 1, ctrl)   # back in the same place: no settling again
    assert ctrl.phase == "APPROACH"
    assert staged(None, 1, staged((200, 150, 280, 236), 3)[0])[1] == MotionCommand()  # default: off


def test_staged_controller_searches_once_towards_the_side_the_target_left():
    cfg = StagedControllerConfig(coast_frames=1, search_turn=0.15, search_frames=3)
    ctrl, command = staged((440, 60, 480, 210), 3, StagedFollowController(cfg))  # far right, turning right
    ctrl, command = staged(None, 2, ctrl)  # brief loss first: still
    assert ctrl.phase == "SEARCH" and command == MotionCommand(turn=-0.15)  # right = negative
    ctrl, command = staged(None, 2, ctrl)
    assert ctrl.phase == "SEARCH"
    ctrl, command = staged(None, 1, ctrl)  # search time over
    assert ctrl.phase == "IDLE" and command == MotionCommand()
    ctrl, command = staged((220, 60, 260, 210), 3, StagedFollowController(cfg))  # lost in the centre
    ctrl, command = staged(None, 3, ctrl)
    assert ctrl.phase == "IDLE" and command == MotionCommand()  # no search
