"""Single source of truth for supported, *matched* Torchreid MSMT17 profiles.

The CLI override affects V5/V5.1 only; historical V2-V4 settings stay intact.
"""
from __future__ import annotations

from pathlib import Path

# Matches checkpoints published in the official Torchreid Model Zoo.
REID_CHECKPOINTS = {
    "osnet_x0_5": "checkpoints/osnet_x0_5_msmt17.pth.tar",
    "osnet_x1_0": "checkpoints/osnet_x1_0_msmt17.pth.tar",
}


def select_reid_model(config: dict, model_name: str | None) -> dict:
    """Return a configuration with a matched architecture/checkpoint override.

    The input is not mutated. With no explicit override, the config is retained.
    """
    if model_name is None:
        return config
    if model_name not in REID_CHECKPOINTS:
        raise ValueError(f"Unsupported Re-ID model: {model_name}")

    updated = dict(config)
    updated["reid"] = dict(config["reid"])
    updated["reid"]["v5"] = {
        **config["reid"]["v5"],
        "model_name": model_name,
        "checkpoint": REID_CHECKPOINTS[model_name],
    }
    return updated


def checkpoint_hint(model_name: str) -> str:
    """Actionable hint without hardcoding a particular architecture."""
    if model_name in REID_CHECKPOINTS:
        return f"python scripts/download_reid_weights.py --model {model_name}_msmt17"
    return "Verify the configured model architecture and matching checkpoint."


def default_output_root(base_root: Path, model_name: str) -> Path:
    """Avoid overwriting existing x0.5 MOT17 benchmarks for x1.0 experiments."""
    return base_root if model_name == "osnet_x0_5" else base_root.with_name(
        f"{base_root.name}_{model_name}"
    )
