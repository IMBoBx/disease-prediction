from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Load models and metadata
disease_model = joblib.load("disease_model.pkl")
severity_model = joblib.load("severity_model.pkl")
label_encoder  = joblib.load("label_encoder.pkl")
all_symptoms   = joblib.load("symptoms_list.pkl")

with open("model_meta.json") as f:
    meta = json.load(f)

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """Return full list of symptoms the model knows about."""
    return jsonify({"symptoms": all_symptoms})

@app.route("/meta", methods=["GET"])
def get_meta():
    """Return model comparison results and metadata."""
    return jsonify(meta)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "symptoms": ["fever", "cough", ...] }
    Returns predicted disease, top-3 probabilities, and severity score.
    """
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "Provide a 'symptoms' list."}), 400

    selected = [s.strip().lower().replace(' ', '_') for s in data["symptoms"]]

    # Build feature vector
    feature_vec = np.array([[1 if s in selected else 0 for s in all_symptoms]])

    # Disease classification
    proba = disease_model.predict_proba(feature_vec)[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {"disease": label_encoder.classes_[i], "confidence": round(float(proba[i]) * 100, 2)}
        for i in top3_idx
    ]

    # Severity regression
    severity_score = float(severity_model.predict(feature_vec)[0])
    severity_score = max(0, round(severity_score, 2))

    if severity_score < 10:
        severity_label = "Mild"
    elif severity_score < 20:
        severity_label = "Moderate"
    else:
        severity_label = "Severe"

    return jsonify({
        "top_predictions": top3,
        "severity_score": severity_score,
        "severity_label": severity_label,
        "model_used": meta["best_model"]
    })

if __name__ == "__main__":
    print(f"Starting server — using {meta['best_model']} ({meta['comparison'][meta['best_model']]['accuracy']}% accuracy)")
    app.run(debug=True, port=5000)