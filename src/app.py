import traceback

from flask import Flask, jsonify, render_template, request

from .config import Config
from .inference.inference import predict
from .model_registry import load_all_models

app = Flask(__name__)
app.config.from_object(Config)

MODELS = load_all_models()


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
    if "file" not in request.files:
        return jsonify({"error": "Файл не передан"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Разрешены только JPG/PNG"}), 400

    model_name = request.form.get("model", Config.DEFAULT_MODEL)
    if model_name not in MODELS:
        return jsonify({"error": f"Неизвестная модель: {model_name}"}), 400

    try:
        image_bytes = file.read()
        class_name, confidence, top3 = predict(
            image_bytes, MODELS[model_name], model_name
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
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
