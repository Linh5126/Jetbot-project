"""CPU-only regression tests for the x0.5 vs x1.0 Re-ID experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

from src.reid.model_config import (REID_CHECKPOINTS, checkpoint_hint,
                                   default_output_root, select_reid_model)
from src.reid.osnet_embedder import OSNetEmbedder


def _config() -> dict:
    with Path("configs/config.yaml").open(encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_default_config_is_unchanged():
    base = _config()
    assert select_reid_model(base, None) is base
    assert base["detector"]["weights"] == "yolov8n.pt"
    assert base["reid"]["v5"]["model_name"] == "osnet_x0_5"


def test_x1_0_override_has_matching_checkpoint_without_mutating_baseline():
    base = _config()
    chosen = select_reid_model(base, "osnet_x1_0")
    assert chosen["reid"]["v5"]["model_name"] == "osnet_x1_0"
    assert chosen["reid"]["v5"]["checkpoint"] == REID_CHECKPOINTS["osnet_x1_0"]
    assert chosen["reid"]["v5"]["embedding_interval"] == base["reid"]["v5"]["embedding_interval"]
    assert chosen["detector"] is base["detector"]
    assert base["reid"]["v5"]["model_name"] == "osnet_x0_5"
    assert base["reid"]["v5"]["checkpoint"] == REID_CHECKPOINTS["osnet_x0_5"]


def test_benchmark_output_isolated_for_x1_0():
    base = Path("outputs/benchmark_final")
    assert default_output_root(base, "osnet_x0_5") == base
    assert default_output_root(base, "osnet_x1_0") == Path("outputs/benchmark_final_osnet_x1_0")


def test_invalid_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        select_reid_model(_config(), "osnet_x9_0")


def test_official_downloader_contains_matching_profile():
    spec = importlib.util.spec_from_file_location("download_reid_weights", "scripts/download_reid_weights.py")
    assert spec and spec.loader
    download = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(download)
    x1 = download.MODELS["osnet_x1_0_msmt17"]
    assert x1["file_id"] == "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x"
    assert x1["architecture"] == "osnet_x1_0"
    assert Path(REID_CHECKPOINTS["osnet_x1_0"]).name == x1["filename"]
    assert "osnet_x1_0_msmt17" in checkpoint_hint("osnet_x1_0")


def test_osnet_adapter_receives_selected_x1_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "osnet_x1_0_msmt17.pth.tar"
    path.write_bytes(b"0" * 500_000)
    captured = {}

    class FakeFeatureExtractor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, crops):
            return np.tile(np.array([3.0, 4.0], dtype=np.float32), (len(crops), 1))

    fake_torchreid = types.ModuleType("torchreid")
    fake_utils = types.ModuleType("torchreid.utils")
    fake_utils.FeatureExtractor = FakeFeatureExtractor
    fake_torchreid.utils = fake_utils
    monkeypatch.setitem(sys.modules, "torchreid", fake_torchreid)
    monkeypatch.setitem(sys.modules, "torchreid.utils", fake_utils)

    embedder = OSNetEmbedder(
        model_name="osnet_x1_0", model_path=path, device="cpu", require_checkpoint=True
    )
    features = embedder([np.ones((64, 32, 3), dtype=np.uint8)])
    assert captured["model_name"] == "osnet_x1_0"
    assert captured["model_path"] == str(path.resolve())
    np.testing.assert_allclose(features, np.array([[0.6, 0.8]], dtype=np.float32))
    assert embedder.checkpoint_loaded


def test_missing_x1_checkpoint_explains_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="osnet_x1_0_msmt17"):
        OSNetEmbedder(model_name="osnet_x1_0", model_path=tmp_path / "not-here.pth.tar",
                      device="cpu", require_checkpoint=True)
