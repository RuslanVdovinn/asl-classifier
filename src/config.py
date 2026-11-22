from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[1]


class Config:
    NUM_CLASSES = 29
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASS_NAMES = (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "del",
        "nothing",
        "space",
    )

    MODELS = {  # noqa: RUF012
        "simplecnn": {
            "path": f"{BASE_DIR}/models/simple_cnn_best.pth",
            "label": "SimpleCNN (baseline)",
            "transform": "simple_cnn",
            "img_size": 200,
            "params": "96K",
            "mean": 0.5068,
            "std": 0.2424,
        },
        "resnet18": {
            "path": f"{BASE_DIR}/models/resnet18_finetuned.pth",
            "label": "ResNet18 (Fine-tuned)",
            "transform": "resnet",
            "img_size": 224,
            "params": "11.2M",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "efficientnet": {
            "path": f"{BASE_DIR}/models/efficientnet_finetuned.pth",
            "label": "EfficientNet-B0 (Fine-tuned)",
            "transform": "efficientnet",
            "img_size": 224,
            "params": "4.0M",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }

    DEFAULT_MODEL = "efficientnet"

    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
    ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png")
