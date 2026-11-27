"""Closed-loop PC simulation of the on-board follower.

    python scripts/sim_follow.py                          # every scenario, both modes, parameter grid
    python scripts/sim_follow.py --scenario crossing --mode guarded --seeds 1 -v

The code under test is the real one: onboard/modes.py (ByteTracker, HandRaiseSelector,
GuardedTargetLock, StagedFollowController), the run.py command path and JetBotHttpDriver
(mix, ramp, TTL), driven tick by tick on a simulated clock. Around it are simulated:
  camera   pinhole, 480x270, people projected as boxes plus shoulder/wrist keypoints,
           box noise, random misses, occlusion by nearer people (hidden when > 65 % covered)
  Re-ID    embeddings drawn per person and view (front/side/back), tuned to the distances
           measured on the robot on 25/11: same person median 0.25, other people median 0.58
  robot    wheel deadzone + first-order motor lag + differential-drive kinematics
Calibrated from the 25/11 floor logs (outputs/robot_03-04.csv vs trace_03-04.csv): the
target moves 6.25 image widths/s per unit of effective wheel difference (|wheel| - 0.06),
4 frames of lag, correlation 0.93-0.95. Forward speed from a 27/11 pulse (30 cm in 2 s at wheel
0.32, ~0.17 m/s). Camera pitch and latency are not measured, so the grid sweeps them.

Not simulated: image quality (light, blur, distance), real detector or Re-ID errors beyond
the drawn distributions, obstacles / collision classifier, wheel slip, battery level.
A PASS here means the logic and the control loop hold over those ranges, not that the
real robot will; keep one rehearsal in the real room.
"""
import argparse
import collections
import itertools
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import onboard.modes as modes  # noqa: E402
from src.following.controller import MotionCommand  # noqa: E402
from src.following.jetbot_driver import DriveConfig, JetBotHttpDriver  # noqa: E402

W, H = 480, 270
FPS = 14.0          # measured on the Nano with the pose engine (27/11)
DT = 0.01
ROBOT_RADIUS, PERSON_RADIUS = 0.10, 0.20


@dataclass(frozen=True)
class World:
    hfov_deg: float = 115.0
    vfov_deg: float = 83.0
    pitch_deg: float = 10.0        # camera tilted up (the floor tests saw the ceiling)
    cam_height: float = 0.10
    yaw_gain: float = 6.25         # image widths / s per unit effective wheel difference (25/11 fit)
    motor_deadzone: float = 0.06   # wheel command that does not turn the wheel (25/11 fit)
    speed_scale: float = 0.28      # 27/11 pulse: 30 cm in 2 s at wheel 0.32 (ramped) -> 0.65 m/s per effective wheel
    track: float = 0.12
    motor_tau: float = 0.12
    latency: float = 0.12          # frame -> command at the driver (stream + inference)
    box_noise_px: float = 2.0
    dropout: float = 0.05


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Person(object):
    def __init__(self, pid, path, raise_hands=None, spin=None, height=1.70):
        self.pid, self.path, self.raise_hands, self.spin, self.height = pid, path, raise_hands, spin, height

    def pos(self, t):
        p = self.path
        if t <= p[0][0]:
            return np.array(p[0][1:], float)
        for (t0, x0, y0), (t1, x1, y1) in zip(p, p[1:]):
            if t <= t1:
                a = (t - t0) / max(t1 - t0, 1e-9)
                return np.array([x0 + a * (x1 - x0), y0 + a * (y1 - y0)])
        return np.array(p[-1][1:], float)

    def heading(self, t, robot_xy):
        """Direction the person faces: walking direction, else towards the robot; spins once if asked."""
        v = self.pos(t + 0.2) - self.pos(t - 0.2)
        face = math.atan2(v[1], v[0]) if np.hypot(*v) > 0.02 else math.atan2(*(robot_xy - self.pos(t))[::-1])
        if self.spin and self.spin[0] <= t <= self.spin[1]:
            face += 2 * math.pi * (t - self.spin[0]) / (self.spin[1] - self.spin[0])
        return face

    def hands_up(self, t):
        return bool(self.raise_hands and self.raise_hands[0] <= t <= self.raise_hands[1])


class Looks(object):
    """Synthetic OSNet embeddings per person and view (0 front, 1 side, 2 back)."""

    def __init__(self, rng, pids, lookalike=None):
        self.rng = rng
        common = _unit(rng.randn(512))
        base = {p: _unit(1.3 * common + _unit(rng.randn(512))) for p in pids}
        if lookalike is not None:  # (pid, like_pid): dresses like the other one
            base[lookalike[0]] = _unit(base[lookalike[1]] + 0.55 * _unit(rng.randn(512)))
        self.views = {p: _unit(base[p][None] + 0.35 * _unit(rng.randn(3, 512))) for p in pids}

    def sample(self, pid, view, mix_with=None, mix=0.0):
        v = self.views[pid][view]
        if mix_with is not None:
            v = _unit((1 - mix) * v + mix * self.views[mix_with][0])
        return _unit(v + 0.024 * math.exp(0.35 * self.rng.randn()) * self.rng.randn(512)).astype(np.float32)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Camera(object):
    def __init__(self, world):
        self.w = world
        self.fx = 0.5 / math.tan(math.radians(world.hfov_deg) / 2)   # normalised focal lengths
        self.fy = 0.5 / math.tan(math.radians(world.vfov_deg) / 2)
        self.pitch = math.radians(world.pitch_deg)

    def project(self, forward, lateral, height):
        """Robot-frame point -> (u, v) in pixels, or None behind the camera."""
        if forward < 0.15:
            return None
        u = 0.5 - self.fx * lateral / forward
        v = 0.5 - self.fy * math.tan(math.atan2(height - self.w.cam_height, forward) - self.pitch)
        return u * W, v * H

    def observe(self, t, pose, people, looks, rng):
        """-> list of dicts (pid, box, score, kpts, emb, quality) for the people the detector reports."""
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        raw = []
        for p in people:
            dx, dy = p.pos(t) - (x, y)
            fwd, lat = c * dx + s * dy, -s * dx + c * dy
            top, bottom = self.project(fwd, lat, p.height), self.project(fwd, lat, 0.0)
            if top is None:
                continue
            half = self.fx * W * 0.22 / fwd
            box = np.array([top[0] - half, top[1], top[0] + half, bottom[1]])
            clipped = np.array([max(box[0], 0), max(box[1], 0), min(box[2], W), min(box[3], H)])
            if clipped[2] - clipped[0] < 0.35 * (box[2] - box[0]) or clipped[3] - clipped[1] < 20:
                continue  # mostly outside the frame
            rel = (p.heading(t, np.array([x, y])) - math.atan2(-dy, -dx) + math.pi) % (2 * math.pi) - math.pi
            view = 0 if abs(rel) < math.pi / 3 else (1 if abs(rel) < 2 * math.pi / 3 else 2)
            kp = np.zeros((17, 3), np.float32)
            wrist_h = p.height + 0.15 if p.hands_up(t) else 0.45 * p.height
            for i, (side, h) in {5: (0.18, 0.82), 6: (-0.18, 0.82)}.items():
                kp[i] = self._kpt(fwd, lat + side, h * p.height)
            for i, side in ((9, 0.25), (10, -0.25)):
                kp[i] = self._kpt(fwd, lat + side, wrist_h)
            raw.append(dict(pid=p.pid, depth=fwd, box=clipped, kpts=kp, view=view))
        out = []
        for r in sorted(raw, key=lambda r: r["depth"]):
            b = r["box"]
            area = (b[2] - b[0]) * (b[3] - b[1])
            covered, occluder = 0.0, None
            for o in out:
                ob = o["box"]
                inter = max(0, min(b[2], ob[2]) - max(b[0], ob[0])) * max(0, min(b[3], ob[3]) - max(b[1], ob[1]))
                if inter / area > covered:
                    covered, occluder = inter / area, o["pid"]
            if covered > 0.65 or rng.rand() < self.w.dropout:
                r["hidden"] = True
                continue
            r["box"] = (b + rng.randn(4) * self.w.box_noise_px).clip([0, 0, 0, 0], [W, H, W, H]).astype(np.float32)
            r["score"] = float(np.clip(0.85 - 0.5 * covered + 0.05 * rng.randn(), 0.15, 0.99))
            r["quality"] = 0.8 * (1 - covered)
            r["emb"] = looks.sample(r["pid"], r["view"], occluder, covered if covered > 0.3 else 0.0)
            if covered > 0.3:
                r["kpts"][:, 2] *= 0.4
            out.append(r)
        return [r for r in out if not r.get("hidden")]

    def _kpt(self, forward, lateral, height):
        uv = self.project(forward, lateral, height)
        if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
            return (0, 0, 0.1)
        return (uv[0], uv[1], 0.9)


SEEN = []  # the current frame's simulated detections, read by the identity stub below


def sim_identity_features(frame, detections, reid):
    """Stands in for modes.add_identity_features: embedding + crop quality of the matching person."""
    for d in detections:
        d["target_id"] = d["track_id"]
        best = max(SEEN, key=lambda r: _iou(r["box"], d["bbox"]), default=None)
        if best is None or _iou(best["box"], d["bbox"]) < 0.3:
            continue
        if d["bbox"][3] - d["bbox"][1] >= 28 and d["confidence"] >= 0.35:
            d["crop_quality"], d["embedding"] = best["quality"], best["emb"]
    return detections


# --- scenarios: robot starts at (0, 0) facing +x; target = pid 0 --------------------------------

def _ritual(x=2.5, y=0.0, until=12.0):
    """Stand still, raise both hands at 1-2.5 s, turn around once at 3.5-11 s (enrolment)."""
    return [(0.0, x, y), (until, x, y)], (1.0, 2.5), (3.5, 11.0)


def _walk(path, start, speed):
    """[(x, y), ...] walked at `speed` m/s from time `start` -> timed waypoints."""
    out, t = [(start,) + tuple(path[0])], start
    for a, b in zip(path, path[1:]):
        t += math.hypot(b[0] - a[0], b[1] - a[1]) / speed
        out.append((t,) + tuple(b))
    return out


def scenario(name, guarded):
    """-> (people, duration, lookalike). In nearest mode nobody needs to raise hands."""
    hold, hands, spin = _ritual()
    def target(rest=()):
        return Person(0, hold + list(rest), hands if guarded else None, spin if guarded else None)
    if name == "stand":
        return [target()], 40.0, None
    if name == "walk_away":
        return [target(_walk([(2.5, 0), (8, 0)], 12, 0.37))], 40.0, None
    if name == "turns":
        return [target(_walk([(2.5, 0), (5, 0), (5, 2.5), (2.5, 3.5)], 12, 0.35))], 45.0, None
    if name == "crossing":  # someone walks between the robot and the target
        return [target(_walk([(2.5, 0), (7, 0)], 12, 0.3)),
                Person(1, _walk([(4.3, -2.5), (4.3, 2.5)], 16.5, 0.71))], 42.0, None
    if name in ("crowd", "lookalike"):  # classroom: bystanders, one walks beside the target for a while
        return [target(_walk([(2.5, 0), (5, 0.5), (6.5, -0.5)], 12, 0.3)),
                Person(1, [(0, 3.0, 1.6)]), Person(2, [(0, 3.2, -1.7)]), Person(3, [(0, 1.6, 2.2)]),
                Person(4, [(0, 4.6, -1.4)]),
                Person(5, [(0, 2.0, -1.2), (14, 2.0, -1.2)] + _walk([(2.6, -0.8), (5, -0.3), (6, -1.5)], 15, 0.3))
                ], 45.0, (5, 0) if name == "lookalike" else None
    if name == "sidestep":  # target leaves the view quickly and does not come back
        return [target(_walk([(2.5, 0), (3.5, 0), (3.5, 4.0), (1.0, 7.0)], 14, 1.2)),
                Person(1, [(0, 4.5, -1.5)])], 40.0, None
    if name == "approach":  # after the robot stopped (~0.9 m), the target steps up to it
        return [target([(24.0, 2.5, 0)] + _walk([(2.5, 0), (1.9, 0)], 24, 0.4)[1:])], 34.0, None
    raise ValueError(name)


SCENARIOS = ("stand", "walk_away", "turns", "crossing", "crowd", "lookalike", "sidestep", "approach")


def run(name, mode_name, world, seed, cfg, verbose=False):
    rng = np.random.RandomState(seed)
    people, duration, lookalike = scenario(name, mode_name == "guarded")
    # the robot (~0.29 m/s) is slower than walking: the target stops and waits; check it arrives
    duration = max(duration, max(p.path[-1][0] for p in people) + 30.0)
    looks = Looks(rng, [p.pid for p in people], lookalike)
    cam = Camera(world)
    if mode_name == "guarded":
        modes.OsnetTrt = lambda path: None
        modes.add_identity_features = sim_identity_features
        mode = modes.GuardedMode(cfg, FPS, H, ROOT)
    else:
        mode = modes.NearestMode(cfg, H)
    clock = [0.0]
    driver = JetBotHttpDriver("http://sim", DriveConfig(**cfg["drive"]), post=lambda path, payload: None,
                              get=None, clock=lambda: clock[0], start_thread=False)
    pose = np.zeros(3)
    wheel = np.zeros(2)                       # actual (effective) wheel speeds
    # yaw_gain is in image widths/s; near the centre d(u)/d(yaw) = 1 / (2 tan(hfov/2)) image widths per rad
    k_wheel = world.yaw_gain * 2 * math.tan(math.radians(world.hfov_deg) / 2) * world.track
    pending, frame_id, next_frame, next_tick = [], 0, 0.0, 0.0
    frame = np.zeros((H, W, 3), np.uint8)
    stats = dict(collision=False, min_gap=99.0, wrong_motion=0, motion_frames=0, follow_t=None,
                 waits=0, fwd_close=0, dist=[], flips=0, lost_frames=0, follow_frames=0,
                 lost_why=collections.Counter())
    last_turn_sign, last_phase = 0, None
    for step in range(int(duration / DT)):
        t = step * DT
        clock[0] = t
        if t >= next_frame:
            next_frame += 1.0 / FPS
            pending.append((t + world.latency, cam.observe(t, pose, people, looks, rng), pose.copy(), t))
        while pending and pending[0][0] <= t:
            _, seen, seen_pose, t_seen = pending.pop(0)
            frame_id += 1
            SEEN[:] = seen
            boxes = np.array([r["box"] for r in seen], np.float32).reshape(-1, 4)
            scores = np.array([r["score"] for r in seen], np.float32)
            kpts = np.array([r["kpts"] for r in seen], np.float32).reshape(-1, 17, 3)
            command, fields = mode.step(frame, frame_id, boxes, scores, kpts, t)
            driver.send(command)
            phase = fields["phase"]
            if phase == "WAIT" and last_phase == "FOLLOW":
                stats["waits"] += 1
            if phase == "FOLLOW" and stats["follow_t"] is None and (mode_name == "guarded" or command != MotionCommand()):
                stats["follow_t"] = t
            last_phase = phase
            moving = command != MotionCommand()
            bbox = fields.get("bbox")
            followed = max(seen, key=lambda r: _iou(r["box"], bbox), default=None) if bbox is not None else None
            if moving:
                stats["motion_frames"] += 1
                if bbox is None:  # no target box: only a short straight COAST or a one-way SEARCH turn
                    stats["wrong_motion"] += fields["reason"] not in ("COAST", "SEARCH")
                elif (followed is None or followed["pid"] != 0 or _iou(followed["box"], bbox) <= 0.0 or (
                        _iou(followed["box"], bbox) < 0.3  # loose fit is fine when it can only be the target
                        and any(r["pid"] != 0 and _iou(r["box"], bbox) > 0 for r in seen))):
                    stats["wrong_motion"] += 1
                    if verbose:
                        print("  wrong motion %.2fs: state %s reason %s guard %s box %s seen %s" % (
                            t, fields["state"], fields["reason"], getattr(getattr(mode, "guard", None),
                            "diagnostics", None) and mode.guard.diagnostics.reason_code,
                            np.round(bbox), [(r["pid"], np.round(r["box"])) for r in seen]))
            target_gap = np.hypot(*(people[0].pos(t) - pose[:2]))
            if command.forward > 0 and target_gap < 0.9:
                stats["fwd_close"] += 1
            if phase == "FOLLOW":
                stats["follow_frames"] += 1
                if fields["state"] in ("LOST", "VERIFYING"):
                    stats["lost_frames"] += 1
                    stats["lost_why"][mode.guard.diagnostics.reason_code] += 1
                stats["dist"].append((t, target_gap))
                sign = int(np.sign(command.turn))
                if sign and last_turn_sign and sign != last_turn_sign:
                    stats["flips"] += 1
                last_turn_sign = sign or last_turn_sign
            if verbose and frame_id % 14 == 0:
                print("%5.1fs %-6s %-10s %-9s gap %.2f cmd %+.2f/%+.2f wheels %+.2f/%+.2f pose %.2f,%.2f,%+.0f" % (
                    t, phase, fields["state"], fields["reason"], target_gap, command.forward, command.turn,
                    driver.last_wheels[0], driver.last_wheels[1], pose[0], pose[1], math.degrees(pose[2])))
        if t >= next_tick:
            next_tick += 1.0 / driver.cfg.send_hz
            driver._tick()
        sent = np.array(driver.last_wheels)
        eff = np.sign(sent) * np.maximum(0.0, np.abs(sent) - world.motor_deadzone)
        wheel += (eff - wheel) * DT / world.motor_tau
        v = world.speed_scale * k_wheel * wheel.mean()
        w = k_wheel * (wheel[1] - wheel[0]) / world.track
        pose[2] += w * DT
        pose[0] += v * math.cos(pose[2]) * DT
        pose[1] += v * math.sin(pose[2]) * DT
        for p in people:
            gap = np.hypot(*(p.pos(t) - pose[:2])) - ROBOT_RADIUS - PERSON_RADIUS
            stats["min_gap"] = min(stats["min_gap"], gap)
            stats["collision"] |= gap < 0.0 and v > 0.02  # contact caused by the robot driving forward
    driver.close()
    stats["final_gap"] = np.hypot(*(people[0].pos(duration) - pose[:2]))
    stats["final_phase"] = last_phase
    return stats


def verdict(name, s):
    """-> list of failed checks (empty = PASS)."""
    bad = []
    if s["collision"]:
        bad.append("collision")
    if s["wrong_motion"]:
        bad.append("moved for someone else x%d" % s["wrong_motion"])
    if s["fwd_close"]:
        bad.append("forward within 0.9 m x%d" % s["fwd_close"])
    if name in ("sidestep", "lookalike"):  # keeping up or stopping are both fine; following someone else is not
        return bad
    if s["follow_t"] is None:
        bad.append("never followed")
    elif name != "approach" and not 0.8 <= s["final_gap"] <= 2.3:
        bad.append("ended %.1f m from target" % s["final_gap"])
    if name != "approach" and s["waits"]:
        bad.append("lost target -> WAIT x%d" % s["waits"])
    return bad


GRID = dict(speed_scale=(0.2, 0.28, 0.4), latency=(0.08, 0.25), pitch_deg=(0.0, 20.0), yaw_gain=(4.5, 8.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "jetbot.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument("--mode", choices=("guarded", "nearest"), action="append")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--nominal", action="store_true", help="only the calibrated World, no grid")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the run once a second")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    worlds = [World()] if args.nominal or args.verbose else [
        replace(World(), **dict(zip(GRID, values))) for values in itertools.product(*GRID.values())]
    failures = 0
    for mode_name in args.mode or ("guarded", "nearest"):
        for name in args.scenario or SCENARIOS:
            results = []
            for world in worlds:
                for seed in range(args.seeds):
                    s = run(name, mode_name, world, seed, cfg, args.verbose)
                    results.append((world, seed, s, verdict(name, s)))
            bad = [r for r in results if r[3]]
            gaps = [r[2]["final_gap"] for r in results]
            follow = [r[2]["follow_t"] for r in results if r[2]["follow_t"] is not None]
            flips = [r[2]["flips"] / max(r[2]["follow_frames"] / FPS, 1) * 10 for r in results]
            lost = [r[2]["lost_frames"] / max(r[2]["follow_frames"], 1) for r in results]
            print("%-8s %-10s %3d/%-3d PASS  final gap %.2f-%.2f m  min gap %.2f m  follow at %s  "
                  "turn flips/10s %.1f  lost %.0f%%" % (
                      mode_name, name, len(results) - len(bad), len(results), min(gaps), max(gaps),
                      min(r[2]["min_gap"] for r in results),
                      "%.1f s" % np.median(follow) if follow else "-", np.median(flips), 100 * np.mean(lost)))
            reasons = {}
            for world, seed, s, why in bad:
                for w in why:
                    reasons.setdefault(w.split(" x")[0].split(" m ")[0], []).append(
                        "speed %.1f lat %.2f pitch %.0f yaw %.1f seed %d" % (
                            world.speed_scale, world.latency, world.pitch_deg, world.yaw_gain, seed))
            why = sum((r[2]["lost_why"] for r in results), collections.Counter())
            if why:
                print("    not verified: " + ", ".join("%s %d%%" % (k, 100 * v / sum(why.values()))
                                                     for k, v in why.most_common(4)))
            for w, where in sorted(reasons.items()):
                print("    FAIL %-32s x%-3d e.g. %s" % (w, len(where), where[0]))
            failures += mode_name == "guarded" and len(bad)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
