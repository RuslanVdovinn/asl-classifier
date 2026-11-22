import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from .config import Config
from .model_registry import load_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate_model(name: str, data_dir: Path) -> dict:
    loaded = load_model(name)
    dataset = datasets.ImageFolder(data_dir, transform=loaded.transform)

    if list(loaded.class_names) != dataset.classes:
        raise SystemExit(f"Порядок классов не совпадает: {dataset.classes}")

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    labels: list[int] = []
    preds: list[int] = []
    top3_hits = 0

    with torch.inference_mode():
        for images, targets in loader:
            logits = loaded.model(images.to(Config.DEVICE))
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(targets.tolist())
            top3 = logits.topk(3, dim=1).indices.cpu()
            top3_hits += (top3 == targets[:, None]).any(dim=1).sum().item()

    total = len(labels)
    meta = Config.MODELS[name]
    all_labels = list(range(len(loaded.class_names)))

    return {
        "n": total,
        "accuracy": accuracy_score(labels, preds),
        "top3_accuracy": top3_hits / total if total else 0.0,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "report": classification_report(
            labels,
            preds,
            labels=all_labels,
            target_names=list(loaded.class_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, preds, labels=all_labels).tolist(),
        "img_size": meta["img_size"],
        "transform": meta["transform"],
        "checkpoint_sha256": _sha256(Path(meta["path"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка моделей на независимом наборе изображений"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=list(Config.MODELS))
    parser.add_argument("--out", default=Path("reports/metrics.json"), type=Path)
    args = parser.parse_args()

    unknown = set(args.models) - set(Config.MODELS)
    if unknown:
        parser.error(f"Неизвестные модели: {sorted(unknown)}")
    if not args.data_dir.is_dir():
        parser.error(f"Нет каталога: {args.data_dir}")

    result = {
        "dataset": str(args.data_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "models": {},
    }

    for name in args.models:
        print(f"Оценка {name}...")
        result["models"][name] = evaluate_model(name, args.data_dir)
        metrics = result["models"][name]
        print(
            f"  n={metrics['n']} "
            f"acc={metrics['accuracy']:.4f} "
            f"top3={metrics['top3_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Записано: {args.out}")


if __name__ == "__main__":
    main()
