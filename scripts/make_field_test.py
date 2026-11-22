import argparse
import json
import shutil
from pathlib import Path

import torch
from torchvision import datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Собрать компактный held-out набор из test-split ASL Alphabet"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/asl_alphabet_train"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/field_test"),
    )
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} уже существует; удали его или укажи другой --out")

    dataset = datasets.ImageFolder(args.data_dir)
    total = len(dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    test_indices = permutation[train_size + val_size:]

    by_class = {name: [] for name in dataset.classes}
    for index in test_indices:
        _, label = dataset.samples[index]
        name = dataset.classes[label]
        if len(by_class[name]) < args.per_class:
            by_class[name].append(index)
        if all(len(indices) >= args.per_class for indices in by_class.values()):
            break

    missing = {
        name: len(indices)
        for name, indices in by_class.items()
        if len(indices) < args.per_class
    }
    if missing:
        raise SystemExit(f"Не хватает изображений: {missing}")

    manifest = {
        "source": str(args.data_dir),
        "seed": args.seed,
        "dataset_size": total,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "per_class": args.per_class,
        "note": (
            "Held-out 15% test split из asl_alphabet_train (seed 42). "
            "Изображения не участвовали в обучении этих чекпоинтов, "
            "но random split не гарантирует независимость от похожих кадров."
        ),
        "files": [],
    }

    for name, indices in by_class.items():
        class_dir = args.out / name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in indices:
            source, _ = dataset.samples[index]
            destination = class_dir / Path(source).name
            shutil.copy2(source, destination)
            manifest["files"].append(
                {
                    "path": str(destination.relative_to(args.out)),
                    "source_index": index,
                    "source": source,
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "README.md").write_text(
        "# field_test\n\n"
        "Компактный held-out набор: по "
        f"{args.per_class} изображения на класс из 15% test-split "
        f"`asl_alphabet_train` (seed {args.seed}).\n\n"
        "Эти изображения не участвовали в обучении чекпоинтов, но random split "
        "не гарантирует независимость от похожих кадров. Для честной оценки "
        "обобщения нужен отдельный источник/другие руки.\n\n"
        "Сгенерировано `scripts/make_field_test.py`.\n",
        encoding="utf-8",
    )

    print(f"Готово: {args.out}")
    print(f"Классов: {len(by_class)}, изображений: {sum(map(len, by_class.values()))}")


if __name__ == "__main__":
    main()
