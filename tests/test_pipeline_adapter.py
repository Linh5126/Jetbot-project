"""Adapter integration checks without Ultralytics, Torchreid or GPU."""
from __future__ import annotations

import numpy as np

from src.runtime.person_pipeline import PersonIdentityPipeline, extract_tracks


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10, 10, 80, 100]])
        self.id = FakeTensor([5])
        self.conf = FakeTensor([0.96])

    def __len__(self):
        return 1


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def track(self, **kwargs):
        return [FakeResult()]


class FakeManager:
    min_box_height = 28.0
    min_embedding_confidence = 0.35

    def has_track(self, track_id):
        return False

    def update(self, frame_id, detections, embeddings):
        self.received = (detections, embeddings)
        return {det["track_id"]: 13 for det in detections}

    def association_of(self, track_id):
        return "new"


class FakeEmbedder:
    def __call__(self, crops):
        return np.ones((len(crops), 8), dtype=np.float32)


def test_tracks_adapter_and_pipeline_passes_quality_metadata():
    assert extract_tracks(FakeResult())[0]["track_id"] == 5
    pipeline = PersonIdentityPipeline.__new__(PersonIdentityPipeline)
    pipeline.mode = "v5_1"
    pipeline.device = "cpu"
    pipeline.confidence = 0.1
    pipeline.detector_cfg = {"classes": [0], "iou": 0.5, "image_size": 640}
    pipeline.tracker_yaml = "mock.yml"
    pipeline.model = FakeModel()
    pipeline.manager = FakeManager()
    pipeline.embedder = FakeEmbedder()
    pipeline.interval = 3
    frame = np.random.default_rng(1).integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
    detections = pipeline.process(frame, 1)
    assert len(detections) == 1
    assert detections[0]["target_id"] == 13
    assert detections[0]["association"] == "new"
    assert 0 <= detections[0]["crop_quality"] <= 1
    assert 5 in pipeline.manager.received[1]


def test_guarded_pipeline_provides_fresh_features_without_changing_manager_cadence():
    class EstablishedManager(FakeManager):
        def has_track(self, track_id):
            return True

    pipeline = PersonIdentityPipeline.__new__(PersonIdentityPipeline)
    pipeline.mode = "v5"
    pipeline.device = "cpu"
    pipeline.confidence = 0.1
    pipeline.detector_cfg = {"classes": [0], "iou": 0.5, "image_size": 640}
    pipeline.tracker_yaml = "mock.yml"
    pipeline.model = FakeModel()
    pipeline.manager = EstablishedManager()
    pipeline.embedder = FakeEmbedder()
    pipeline.interval = 3
    pipeline.guard_features = True
    frame = np.random.default_rng(2).integers(0, 255, size=(120, 120, 3), dtype=np.uint8)

    detections = pipeline.process(frame, 2)  # Not an ordinary gallery update frame.
    assert "embedding" in detections[0]
    assert "crop_quality" in detections[0]
    assert pipeline.manager.received[1] == {}  # V5 gallery update cadence preserved.
    detections = pipeline.process(frame, 3)
    assert 5 in pipeline.manager.received[1]
