import io

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from ..config import Config
from ..model_registry import LoadedModel


def predict(image_bytes: bytes, loaded: LoadedModel):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Не удалось открыть изображение: {e}")

    tensor = loaded.transform(image).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        outputs = loaded.model(tensor)
        probs = F.softmax(outputs, dim=1)

    top3_prob, top3_idx = torch.topk(probs, k=3, dim=1)
    top3_prob = top3_prob.cpu().numpy()[0]
    top3_idx = top3_idx.cpu().numpy()[0]

    results = [
        {"class": loaded.class_names[idx], "probability": float(prob)}
        for idx, prob in zip(top3_idx, top3_prob)
    ]
    return results[0]["class"], results[0]["probability"], results
