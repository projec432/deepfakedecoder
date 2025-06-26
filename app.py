from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and scaler
with open("xgb_baseline.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    file_path = "temp_audio.wav"
    audio_file.save(file_path)

    try:
        # Load and preprocess audio
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        scaled = scaler.transform([mfcc_mean])
        prediction = model.predict(scaled)

        return jsonify({"prediction": str(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
