import logging
import sys
import time
import traceback

from flask import Flask, jsonify, render_template, request

from .config import Config
from .inference.inference import predict
from .model_registry import ModelStore

DEBUG = "--debug" in sys.argv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.extensions["model_store"] = ModelStore()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    models_meta = [{"name": k, **v} for k, v in Config.MODELS.items()]
    return render_template(
        "index.html", models=models_meta, default=Config.DEFAULT_MODEL
    )


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    t_start = time.perf_counter()
    if "file" not in request.files:
        logger.warning("Запрос без файла")
        return jsonify({"success": False, "error": "Файл не передан"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.warning("Пустое имя файла")
        return jsonify({"success": False, "error": "Файл не выбран"}), 400

    if not allowed_file(file.filename):
        logger.warning(f"Недопустимое расширение: {file.filename}")
        return jsonify(
            {
                "success": False,
                "error": f"Разрешены только: {', '.join(Config.ALLOWED_EXTENSIONS)}",
            }
        ), 400

    model_name = request.form.get("model", Config.DEFAULT_MODEL)
    if model_name not in Config.MODELS:
        logger.warning(f"Неизвестная модель: {model_name}")
        return jsonify(
            {
                "success": False,
                "error": f"Неизвестная модель: {model_name}. "
                f"Доступны: {list(Config.MODELS.keys())}",
            }
        ), 400

    try:
        image_bytes = file.read()
        model = app.extensions["model_store"].get(model_name)
        t_infer = time.perf_counter()
        class_name, confidence, top3 = predict(image_bytes, model, model_name)
        infer_ms = (time.perf_counter() - t_infer) * 1000
    except Exception as e:
        logger.exception(f"Ошибка инференса (модель={model_name})")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Ошибка инференса: {e}"}), 500

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"predict | model={model_name} | file={file.filename} | "
        f"size={len(image_bytes)}B | pred={class_name} | "
        f"conf={confidence:.4f} | infer={infer_ms:.1f}ms | total={total_ms:.1f}ms"
    )
    return jsonify(
        {
            "success": True,
            "model": model_name,
            "prediction": class_name,
            "confidence": round(confidence, 4),
            "top3": top3,
        }
    )


if __name__ == "__main__":
    app.run(port=5000, debug=DEBUG)
