from __future__ import annotations

import argparse
from pathlib import Path

MODELS = {
    # Torchreid official Model Zoo: cross-domain models trained on MSMT17.
    "osnet_x0_5_msmt17": {
        "file_id": "1DHgmb6XV4fwG3n-CnCM0zdL9nMsZ9_RF",
        "filename": "osnet_x0_5_msmt17.pth.tar",
        "url": "https://drive.google.com/file/d/1DHgmb6XV4fwG3n-CnCM0zdL9nMsZ9_RF/view",
        "architecture": "osnet_x0_5",
        "training_data": "MSMT17 combineall",
    },
    "osnet_x1_0_msmt17": {
        "file_id": "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x",
        "filename": "osnet_x1_0_msmt17.pth.tar",
        "url": "https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view",
        "architecture": "osnet_x1_0",
        "training_data": "MSMT17 combineall",
    },

}


def main():
    parser = argparse.ArgumentParser(description="Download official Torchreid MSMT17 OSNet weights")
    parser.add_argument("--model", choices=sorted(MODELS), default="osnet_x0_5_msmt17")
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Thiếu gdown. Chạy: python -m pip install gdown") from exc

    spec = MODELS[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / spec["filename"]

    if output_path.exists() and output_path.stat().st_size >= 500_000:
        print(f"Checkpoint đã tồn tại: {output_path} ({output_path.stat().st_size:,} bytes)")
        return

    print(f"Downloading {args.model} ({spec['training_data']}) ...")
    downloaded = gdown.download(id=spec["file_id"], output=str(output_path), quiet=False)
    if not downloaded or not output_path.exists() or output_path.stat().st_size < 500_000:
        output_path.unlink(missing_ok=True)
        raise SystemExit(
            "Tải checkpoint thất bại hoặc file không hợp lệ. "
            f"Bạn có thể tải thủ công từ: {spec['url']}"
        )

    print(f"Saved: {output_path.resolve()}")
    print(f"Size: {output_path.stat().st_size:,} bytes")
    print(f"Architecture: {spec['architecture']}")
    print("Hãy giữ nguyên file checkpoint để bảo đảm tái lập thí nghiệm.")


if __name__ == "__main__":
    main()
