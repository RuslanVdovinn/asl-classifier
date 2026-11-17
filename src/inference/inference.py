import io

import torch
import torch.nn.functional as F
from PIL import Image

from config import Config
from data.transforms import get_transforms

transforms = {
    "simplecnn": get_transforms(
        model_type="simple_cnn",
        img_size=Config.IMG_SIZE,
        mean=Config.MODELS["simplecnn"]["mean"],
        std=Config.MODELS["simplecnn"]["std"],
    )[1],
    "resnet18": get_transforms(model_type="resnet")[1],
    "efficientnet": get_transforms(model_type="efficientnet")[1],
}


def predict(image_bytes: bytes, model, model_name):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transforms[model_name](image).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)

    top3_prob, top3_idx = torch.topk(probs, k=3, dim=1)
    top3_prob = top3_prob.cpu().numpy()[0]
    top3_idx = top3_idx.cpu().numpy()[0]

    results = [
        {"class": Config.CLASS_NAMES[idx], "probability": float(prob)}
        for idx, prob in zip(top3_idx, top3_prob)
    ]
    return results[0]["class"], results[0]["probability"], results
