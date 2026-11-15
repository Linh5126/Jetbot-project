from __future__ import annotations

from pathlib import Path

from src.reid.model_config import checkpoint_hint

import cv2
import numpy as np


class OSNetEmbedder:
    """Torchreid FeatureExtractor adapter with explicit checkpoint support.

    For V2-V4, model_path may be omitted to preserve the historical default
    ImageNet initialization. V5 passes require_checkpoint=True so it can never
    silently fall back to ImageNet weights.
    """

    def __init__(
        self,
        model_name="osnet_x0_25",
        device="cuda",
        image_size=(256, 128),
        model_path=None,
        require_checkpoint=False,
    ):
        checkpoint = Path(model_path).expanduser() if model_path else None
        if require_checkpoint:
            if checkpoint is None:
                raise FileNotFoundError("V5 bắt buộc cấu hình reid.v5.checkpoint.")
            if not checkpoint.is_file() or checkpoint.stat().st_size < 500_000:
                raise FileNotFoundError(
                    f"Không tìm thấy checkpoint Person Re-ID hợp lệ: {checkpoint}\n"
                    "Chạy: " + checkpoint_hint(str(model_name))
                )

        try:
            from torchreid.utils import FeatureExtractor
        except Exception as exc:
            raise RuntimeError(
                "Không import được Torchreid. Cài theo README.md: "
                "pip install torchreid gdown"
            ) from exc

        import torch

        if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
            if require_checkpoint:
                raise RuntimeError(
                    "V5 yêu cầu CUDA để benchmark trên cấu hình đã chọn, "
                    "nhưng PyTorch không nhận GPU. Kiểm tra bản PyTorch CUDA và driver."
                )
            device = "cpu"

        self.model_name = str(model_name)
        self.model_path = str(checkpoint.resolve()) if checkpoint else None
        self.device = str(device)
        self.image_size = tuple(int(x) for x in image_size)
        self.extractor = FeatureExtractor(
            model_name=self.model_name,
            model_path=self.model_path or "",
            image_size=self.image_size,
            device=self.device,
        )
        self.checkpoint_loaded = bool(self.model_path)
        print(
            f"[ReID] model={self.model_name} | device={self.device} | "
            f"checkpoint={self.model_path or 'Torchreid default pretrained'}"
        )

    def __call__(self, crops):
        prepared = []
        target_h, target_w = self.image_size
        for crop in crops:
            if crop is None or crop.size == 0:
                prepared.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (target_w, target_h))
            prepared.append(rgb)

        if not prepared:
            return np.empty((0, 0), dtype=np.float32)

        features = self.extractor(prepared)
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        features = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-12)
