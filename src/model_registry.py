import threading

import torch
from torch import nn
from torchvision import models

from .config import Config
from .models.simple_cnn import SimpleCNN


class ModelLoadError(RuntimeError):
    pass


def build_resnet18(num_classes=29):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes=29):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


BUILDERS = {
    "simplecnn": SimpleCNN,
    "resnet18": build_resnet18,
    "efficientnet": build_efficientnet_b0,
}


def load_model(name: str):
    meta = Config.MODELS[name]
    model = BUILDERS[name](Config.NUM_CLASSES)

    try:
        checkpoint = torch.load(
            meta["path"], map_location=Config.DEVICE, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError as exc:
        raise ModelLoadError(
            f"Веса модели '{name}' не найдены: {meta['path']}"
        ) from exc
    except Exception as exc:
        raise ModelLoadError(f"Не удалось загрузить модель '{name}': {exc}") from exc

    print(f"Загружена модель '{name}' ({meta['label']})")
    return model.to(Config.DEVICE).eval()


class ModelStore:
    def __init__(self, loader=load_model):
        self._loader = loader
        self._models = {}
        self._lock = threading.Lock()

    def get(self, name: str):
        model = self._models.get(name)
        if model is not None:
            return model

        with self._lock:
            model = self._models.get(name)
            if model is None:
                model = self._loader(name)
                self._models[name] = model

        return model
