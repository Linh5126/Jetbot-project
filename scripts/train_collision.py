"""Train the free/blocked obstacle classifier (ResNet18) and export ONNX.

    python scripts/train_collision.py            # data/collision/{blocked,free}/*.jpg

Writes checkpoints/collision_resnet18.pth and models/onboard/collision_resnet18.onnx
(static 1x3x224x224, opset 11 for TensorRT 7.1). Copy the ONNX to the Nano, then
`sh onboard/build_engines.sh` builds collision_resnet18_fp16.engine.
Preprocessing must match onboard/models.py: full frame -> RGB, resize 224x224, ImageNet normalise.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def loaders(data: Path, batch: int, seed: int):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomHorizontalFlip(),  # left/right obstacles are symmetric
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    val_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize(MEAN, STD)])
    train_set, val_set = datasets.ImageFolder(data, train_tf), datasets.ImageFolder(data, val_tf)
    if train_set.classes != ["blocked", "free"]:
        raise SystemExit(f"Expected folders blocked/ and free/ in {data}, got {train_set.classes}")
    order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(1, len(order) // 5)
    return (DataLoader(Subset(train_set, order[n_val:]), batch, shuffle=True, num_workers=2),
            DataLoader(Subset(val_set, order[:n_val]), batch, num_workers=2), train_set.targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(ROOT / "data" / "collision"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, targets = loaders(Path(args.data), args.batch, args.seed)
    counts = [targets.count(0), targets.count(1)]
    print(f"blocked {counts[0]}, free {counts[1]} images; device {device}")
    if min(counts) < 30:
        raise SystemExit("Need at least 30 images per class (100+ recommended)")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ckpt = ROOT / "checkpoints" / "collision_resnet18.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train:
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        model.eval()
        # Missing an obstacle is the costly error, so report blocked recall too.
        hit = total = tp = pos = 0
        with torch.no_grad():
            for x, y in val:
                pred = model(x.to(device)).argmax(1).cpu()
                hit += int((pred == y).sum())
                total += len(y)
                tp += int(((pred == 0) & (y == 0)).sum())
                pos += int((y == 0).sum())
        acc = hit / total
        print(f"epoch {epoch + 1}: val acc {acc:.3f}, blocked recall {tp / max(pos, 1):.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.cpu().eval()
    onnx = ROOT / "models" / "onboard" / "collision_resnet18.onnx"
    onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(onnx), opset_version=11,
                      input_names=["input"], output_names=["logits"], dynamo=False)
    print(f"best val acc {best:.3f}; saved {ckpt} and {onnx}")


if __name__ == "__main__":
    main()
