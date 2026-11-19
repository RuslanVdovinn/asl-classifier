import torch
from torch import nn
from torchvision import models

from .config import Config
from .models.simple_cnn import SimpleCNN


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


def load_all_models():
    loaded = {}
    for name, meta in Config.MODELS.items():
        builder = BUILDERS[name]
        model = builder(Config.NUM_CLASSES)

        checkpoint = torch.load(
            meta["path"], map_location=Config.DEVICE, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(Config.DEVICE).eval()

        loaded[name] = model
        print(
            f"Загружена модель '{name}' ({meta['label']}), "
            f"params={meta['params']}, test_acc={meta['test_acc']}"
        )
    return loaded
