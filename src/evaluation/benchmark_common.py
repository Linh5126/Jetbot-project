"""Shared MOT17 benchmark helpers; no model inference."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def get_fps(seq_dir: Path, fallback: float) -> float:
    info = seq_dir / "seqinfo.ini"
    if info.exists():
        for line in info.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("framerate"):
                try:
                    return float(line.split("=", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
    return float(fallback)



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def write_mot_row(writer, frame_id: int, identity_id: int, bbox, confidence: float):
    x1, y1, x2, y2 = bbox
    writer.writerow([
        frame_id, identity_id,
        f"{x1:.2f}", f"{y1:.2f}",
        f"{x2 - x1:.2f}", f"{y2 - y1:.2f}",
        f"{confidence:.4f}", -1, -1, -1
    ])

