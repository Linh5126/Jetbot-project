from pathlib import Path


def discover_sequences(root, split="train", suffix="FRCNN"):
    """
    Expected layout:
      data/datasets/MOT17/train/MOT17-02-FRCNN/img1/000001.jpg
    """
    root = Path(root)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Không thấy thư mục dataset: {split_dir}\n"
            "Cần đặt MOT17 theo dạng data/datasets/MOT17/train/MOT17-XX-FRCNN/img1"
        )
    found = []
    for folder in split_dir.iterdir():
        if not folder.is_dir() or not (folder / "img1").is_dir():
            continue
        if suffix and not folder.name.endswith("-" + suffix):
            continue
        found.append(folder)
    return sorted(found, key=lambda p: p.name)


def read_mot_gt(gt_path, min_visibility=0.0):
    """
    MOT17 gt.txt columns:
    frame,id,x,y,w,h,conf,class,visibility
    Keeps pedestrian class=1 with conf>0.
    """
    rows = []
    gt_path = Path(gt_path)
    if not gt_path.exists():
        return rows
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = [v.strip() for v in line.split(",")]
            if len(fields) < 7:
                continue
            try:
                values = [float(v) for v in fields]
            except ValueError:
                continue
            frame_id, identity = int(values[0]), int(values[1])
            x, y, w, h, conf = values[2:7]
            cls = int(values[7]) if len(values) > 7 else 1
            visibility = values[8] if len(values) > 8 else 1.0
            if cls == 1 and conf > 0 and visibility >= min_visibility and w > 0 and h > 0:
                rows.append((frame_id, identity, [x, y, x + w, y + h]))
    return rows


def read_mot_result(result_path):
    rows = []
    result_path = Path(result_path)
    if not result_path.exists():
        return rows
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = [v.strip() for v in line.split(",")]
            if len(fields) < 7:
                continue
            frame_id, identity = int(float(fields[0])), int(float(fields[1]))
            x, y, w, h = map(float, fields[2:6])
            rows.append((frame_id, identity, [x, y, x + w, y + h]))
    return rows
