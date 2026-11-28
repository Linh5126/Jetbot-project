"""Minimal ByteTrack tracker (numpy + scipy) for the Nano's Python 3.6.

Parameters from configs/bytetrack_custom.yaml, the same file the PC pipeline uses:
two-stage association (high-score, then low-score detections), constant-velocity
Kalman filter on (cx, cy, aspect, height), track buffer scaled to frame_rate=30.
tests/test_onboard.py checks it gives the same ids and boxes as the PC tracker.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class KalmanXYAH(object):
    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.w_pos, self.w_vel = 1.0 / 20, 1.0 / 160

    def initiate(self, xyah):
        mean = np.r_[xyah, np.zeros(4)]
        h = xyah[3]
        std = [2 * self.w_pos * h, 2 * self.w_pos * h, 1e-2, 2 * self.w_pos * h,
               10 * self.w_vel * h, 10 * self.w_vel * h, 1e-5, 10 * self.w_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.w_pos * h, self.w_pos * h, 1e-2, self.w_pos * h,
               self.w_vel * h, self.w_vel * h, 1e-5, self.w_vel * h]
        return self.F.dot(mean), self.F.dot(cov).dot(self.F.T) + np.diag(np.square(std))

    def update(self, mean, cov, xyah):
        h = mean[3]
        r = np.diag(np.square([self.w_pos * h, self.w_pos * h, 1e-1, self.w_pos * h]))
        s = self.H.dot(cov).dot(self.H.T) + r
        gain = np.linalg.solve(s, self.H.dot(cov)).T  # cov H^T S^-1 (S, cov symmetric)
        mean = mean + gain.dot(xyah - self.H.dot(mean))
        return mean, cov - gain.dot(s).dot(gain.T)


def _xyah(box):
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w / max(h, 1e-6), h], np.float64)


class Track(object):
    def __init__(self, box, score, kpts=None):
        self.box = np.asarray(box, np.float64)
        self.score = float(score)
        self.kpts = kpts  # 17x3 keypoints of the latest matched detection (pose model)
        self.mean = self.cov = None
        self.state, self.activated = NEW, False
        self.track_id = 0
        self.frame_id = self.start_frame = 0

    @property
    def xyxy(self):
        if self.mean is None:
            return self.box.copy()
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def activate(self, kf, frame_id, track_id):
        self.track_id = track_id
        self.mean, self.cov = kf.initiate(_xyah(self.box))
        self.state = TRACKED
        self.activated = frame_id == 1  # only a track born on frame 1 is confirmed at once; others on their 2nd hit
        self.frame_id = self.start_frame = frame_id

    def update(self, kf, det, frame_id):
        self.mean, self.cov = kf.update(self.mean, self.cov, _xyah(det.box))
        self.score, self.state, self.activated = det.score, TRACKED, True
        self.kpts = det.kpts
        self.frame_id = frame_id


def _iou_cost(tracks, dets):
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)))
    a = np.array([t.xyxy for t in tracks])
    b = np.array([d.box for d in dets])
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return 1.0 - inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _fuse_score(cost, dets):
    if cost.size == 0:
        return cost
    return 1.0 - (1.0 - cost) * np.array([d.score for d in dets])[None, :]


def _assign(cost, thresh):
    """-> matches [(row, col)], unmatched rows, unmatched cols (cost <= thresh)."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    used_r = set(r for r, _ in matches)
    used_c = set(c for _, c in matches)
    return (matches, [r for r in range(cost.shape[0]) if r not in used_r],
            [c for c in range(cost.shape[1]) if c not in used_c])


class ByteTracker(object):
    def __init__(self, track_high_thresh=0.25, track_low_thresh=0.10, new_track_thresh=0.25,
                 track_buffer=60, match_thresh=0.80, fuse_score=True, frame_rate=30, **_):
        self.high, self.low, self.new = track_high_thresh, track_low_thresh, new_track_thresh
        self.match, self.fuse = match_thresh, fuse_score
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.kf = KalmanXYAH()
        self.tracked, self.lost = [], []
        self.frame_id = 0
        self._next_id = 0

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    def update(self, boxes, scores, kpts=None):
        """boxes Nx4 xyxy, scores N, optional keypoints Nx17x3 -> list of
        {track_id, bbox, confidence, kpts} (confirmed tracks; kpts None if not given
        or not matched this frame)."""
        self.frame_id += 1
        fid = self.frame_id
        if kpts is None or len(kpts) != len(boxes):
            kpts = [None] * len(boxes)
        dets = [Track(b, s, k) for b, s, k in zip(boxes, scores, kpts)]
        high = [d for d in dets if d.score >= self.high]
        second = [d for d in dets if self.low < d.score < self.high]
        unconfirmed = [t for t in self.tracked if not t.activated]
        confirmed = [t for t in self.tracked if t.activated]

        # 1st association: confirmed + lost tracks vs high-score detections.
        pool = confirmed + [t for t in self.lost if t not in confirmed]
        for t in pool:
            if t.state != TRACKED:
                t.mean[7] = 0.0  # a lost track keeps its height
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
        cost = _iou_cost(pool, high)
        if self.fuse:
            cost = _fuse_score(cost, high)
        matches, u_track, u_det = _assign(cost, self.match)
        activated, refound, lost, removed = [], [], [], []
        for r, c in matches:
            track = pool[r]
            (activated if track.state == TRACKED else refound).append(track)
            track.update(self.kf, high[c], fid)

        # 2nd association: still-tracked leftovers vs low-score detections.
        leftovers = [pool[i] for i in u_track if pool[i].state == TRACKED]
        matches, u_left, _ = _assign(_iou_cost(leftovers, second), 0.5)
        for r, c in matches:
            leftovers[r].update(self.kf, second[c], fid)
            activated.append(leftovers[r])
        for i in u_left:
            leftovers[i].state = LOST
            lost.append(leftovers[i])

        # Unconfirmed (one-hit) tracks vs remaining high detections.
        rest = [high[i] for i in u_det]  # one-hit tracks are matched without a prediction step
        cost = _iou_cost(unconfirmed, rest)
        if self.fuse:
            cost = _fuse_score(cost, rest)
        matches, u_unconf, u_rest = _assign(cost, 0.7)
        for r, c in matches:
            unconfirmed[r].update(self.kf, rest[c], fid)
            activated.append(unconfirmed[r])
        for i in u_unconf:
            unconfirmed[i].state = REMOVED
            removed.append(unconfirmed[i])

        # New tracks.
        for i in u_rest:
            if rest[i].score >= self.new:
                rest[i].activate(self.kf, fid, self._new_id())
                activated.append(rest[i])

        for t in self.lost:
            if t.state == LOST and fid - t.frame_id > self.max_time_lost:
                t.state = REMOVED
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked += [t for t in activated + refound if t not in self.tracked]
        self.lost = [t for t in self.lost + lost if t.state == LOST and t not in self.tracked]
        self._drop_duplicates()
        return [{"track_id": t.track_id, "bbox": [float(v) for v in t.xyxy],
                 "confidence": t.score, "kpts": t.kpts if t.frame_id == fid else None}
                for t in self.tracked if t.activated]

    def _drop_duplicates(self):
        cost = _iou_cost(self.tracked, [_Box(t.xyxy) for t in self.lost])
        drop_tracked, drop_lost = set(), set()
        for p, q in zip(*np.where(cost < 0.15)):
            a, b = self.tracked[p], self.lost[q]
            if a.frame_id - a.start_frame > b.frame_id - b.start_frame:
                drop_lost.add(q)
            else:
                drop_tracked.add(p)
        self.tracked = [t for i, t in enumerate(self.tracked) if i not in drop_tracked]
        self.lost = [t for i, t in enumerate(self.lost) if i not in drop_lost]


class _Box(object):
    def __init__(self, box):
        self.box = box
