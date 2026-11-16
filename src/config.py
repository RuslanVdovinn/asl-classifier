import torch


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

    IMG_SIZE = 200

    MODELS = {  # noqa: RUF012
        "simplecnn": {
            "path": "../models/simple_cnn_best.pth",
            "label": "SimpleCNN (baseline)",
            "params": "96K",
            "test_acc": "96.28%",
            "mean": 0.5068,
            "std": 0.2424,
        },
        "resnet18": {
            "path": "../models/resnet18_finetuned.pth",
            "label": "ResNet18 (Fine-tuned)",
            "params": "11.2M",
            "test_acc": "99.98%",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "efficientnet": {
            "path": "../models/efficientnet_finetuned.pth",
            "label": "EfficientNet-B0 (Fine-tuned)",
            "params": "4.0M",
            "test_acc": "99.98%",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }

    DEFAULT_MODEL = "efficientnet"

    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
    ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png")
