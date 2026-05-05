print("THIS IS MY APP FILE")

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import speech_recognition as sr
import cv2
import numpy as np
import base64
from pydub import AudioSegment
import tempfile
from tensorflow.keras.models import load_model

# ------------------- SAFE TRANSFORMER IMPORT -------------------
try:
    from transformers import pipeline
    TRANSFORMER_AVAILABLE = True
except:
    TRANSFORMER_AVAILABLE = False


# ------------------- PATH SETUP -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


# ------------------- APP INIT -------------------
app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_DIR, "web"),
    static_url_path=""
)
CORS(app)


# ------------------- GLOBAL STATE -------------------
latest_results = {
    "face": None,
    "text": None,
    "voice": None
}


# ------------------- LOAD MODELS -------------------

# FACE MODEL
face_model = load_model(
    os.path.join(PROJECT_DIR, "models", "emotion_model.h5"),
    compile=False
)

emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ------------------- TEXT MODEL -------------------
text_emotion = None

if TRANSFORMER_AVAILABLE:
    try:
        text_emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        print("✅ Text model loaded successfully")
    except Exception as e:
        print("❌ Text model error:", str(e))


print("✅ Models Loaded")


# ------------------- LABEL NORMALIZATION FIX -------------------
def normalize_label(label):
    label = label.lower()
    mapping = {
        "joy": "happy",
        "anger": "angry",
        "sadness": "sad",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral",
        "disgust": "disgust"
    }
    return mapping.get(label, label)


# ------------------- ROUTES -------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ------------------- FACE -------------------
@app.route("/predict-face", methods=["POST"])
def predict_face():
    try:
        data = request.json["image"]

        img_bytes = base64.b64decode(data.split(",")[1])
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)

        results = []

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48)) / 255.0
            roi = roi.reshape(1,48,48,1)

            pred = face_model.predict(roi, verbose=0)
            emotion = emotion_labels[np.argmax(pred)]

            latest_results["face"] = emotion

            results.append({
                "emotion": emotion,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------- TEXT -------------------
@app.route("/predict-text", methods=["POST"])
def predict_text():
    try:
        text = request.json.get("text", "").strip()

        if not text:
            return jsonify({"emotion": "No Input", "confidence": 0})

        if text_emotion is None:
            return jsonify({"emotion": "Model Error", "confidence": 0})

        preds = text_emotion(text)[0]
        top = max(preds, key=lambda x: x["score"])

        emotion = normalize_label(top["label"])
        confidence = round(top["score"] * 100, 2)

        latest_results["text"] = emotion

        print("TEXT:", text)
        print("TEXT EMOTION:", emotion)

        return jsonify({
            "emotion": emotion,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------- VOICE -------------------
@app.route("/predict-voice", methods=["POST"])
def predict_voice():
    try:
        if text_emotion is None:
            return jsonify({"text": "", "emotion": "Model Error", "confidence": 0})

        file = request.files["audio"]

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        file.save(temp_input.name)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

        audio = AudioSegment.from_file(temp_input.name)
        audio.export(temp_wav.name, format="wav")

        recognizer = sr.Recognizer()

        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
        except:
            text = ""

        if text == "":
            return jsonify({
                "text": "",
                "emotion": "No Speech Detected",
                "confidence": 0
            })

        preds = text_emotion(text)[0]
        top = max(preds, key=lambda x: x["score"])

        emotion = normalize_label(top["label"])
        confidence = round(top["score"] * 100, 2)

        latest_results["voice"] = emotion

        print("VOICE TEXT:", text)
        print("VOICE EMOTION:", emotion)

        return jsonify({
            "text": text,
            "emotion": emotion,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------- FINAL EMOTION -------------------
@app.route("/final-emotion", methods=["GET"])
def final_emotion():
    try:
        weights = {"face": 0.4, "text": 0.3, "voice": 0.3}
        scores = {}

        for source, emotion in latest_results.items():
            if emotion is None:
                continue

            emotion = normalize_label(emotion)
            scores[emotion] = scores.get(emotion, 0) + weights[source]

        if not scores:
            return jsonify({"error": "No data yet"}), 400

        final = max(scores, key=scores.get)

        return jsonify({
            "final_emotion": final,
            "confidence": round(scores[final] * 100, 2),
            "details": latest_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------- RUN -------------------
if __name__ == "__main__":
    app.run(port=5001, debug=True)