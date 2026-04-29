from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import speech_recognition as sr
import cv2
import numpy as np
import base64
from pydub import AudioSegment
import tempfile
import os
from tensorflow.keras.models import load_model
from transformers import pipeline

latest_results = {
    "face": None,
    "text": None,
    "voice": None
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_DIR, "web"),
    static_url_path=""
)
CORS(app)


face_model = load_model(os.path.join(PROJECT_DIR, "models", "emotion_model.h5"))
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

text_emotion = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

print("✅ Models Loaded")



@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")



@app.route("/predict-face", methods=["POST"])
def predict_face():
    try:
        data = request.json["image"]

        img_bytes = base64.b64decode(data.split(",")[1])
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        print("Faces detected:", len(faces))  

        results = []

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]

            roi = cv2.resize(roi, (48,48))
            roi = roi / 255.0
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
        print("FACE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route("/predict-text", methods=["POST"])
def predict_text():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        preds = text_emotion(text)[0]
        top = max(preds, key=lambda x: x["score"])

        emotion_map = {
            "joy": "Happy",
            "sadness": "Sad",
            "anger": "Angry",
            "fear": "Fear",
            "surprise": "Surprise",
            "neutral": "Neutral",
            "disgust": "Disgust"
        }

        emotion = emotion_map.get(top["label"].lower(), "Neutral")
        confidence = round(top["score"] * 100, 2)
        latest_results["text"] = emotion

        return jsonify({
            "emotion": emotion,
            "confidence": confidence
        })

    except Exception as e:
        print("TEXT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route("/predict-voice", methods=["POST"])
def predict_voice():
    try:
        file = request.files["audio"]

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        file.save(temp_input.name)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

        audio = AudioSegment.from_file(temp_input.name, format="webm")
        audio.export(temp_wav.name, format="wav")

        recognizer = sr.Recognizer()

        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        preds = text_emotion(text)[0]
        top = max(preds, key=lambda x: x["score"])

        emotion_map = {
            "joy": "Happy",
            "sadness": "Sad",
            "anger": "Angry",
            "fear": "Fear",
            "surprise": "Surprise"
        }

        emotion = emotion_map.get(top["label"].lower(), "Neutral")
        confidence = round(top["score"] * 100, 2)
        latest_results["voice"] = emotion

        return jsonify({
            "text": text,
            "emotion": emotion,
            "confidence": confidence
        })

    except Exception as e:
        print("VOICE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/final-emotion", methods=["GET"])
def final_emotion():
    try:
        weights = {
            "face": 0.4,
            "text": 0.3,
            "voice": 0.3
        }

        scores = {}

        for source, emotion in latest_results.items():
            if emotion is None:
                continue

            weight = weights[source]

            if emotion not in scores:
                scores[emotion] = 0

            scores[emotion] += weight

        if not scores:
            return jsonify({"error": "No data yet"}), 400

        final_emotion = max(scores, key=scores.get)
        confidence = round(scores[final_emotion] * 100, 2)

        return jsonify({
            "final_emotion": final_emotion,
            "confidence": confidence,
            "details": latest_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-all", methods=["POST"])
def predict_all():
    try:
        results = {}

        text = request.form.get("text", "")
        if text:
            preds = text_emotion(text)[0]
            top = max(preds, key=lambda x: x["score"])
            results["text"] = top["label"]

        if "image" in request.files:
            file = request.files["image"]
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x,y,w,h) = faces[0]
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48,48)) / 255.0
                roi = roi.reshape(1,48,48,1)

                pred = face_model.predict(roi, verbose=0)
                results["face"] = emotion_labels[np.argmax(pred)]

        if "audio" in request.files:
            file = request.files["audio"]

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            file.save(temp.name)

            recognizer = sr.Recognizer()
            with sr.AudioFile(temp.name) as source:
                audio_data = recognizer.record(source)

            text_audio = recognizer.recognize_google(audio_data)

            preds = text_emotion(text_audio)[0]
            top = max(preds, key=lambda x: x["score"])

            results["voice"] = top["label"]

        weights = {"face": 0.4, "text": 0.3, "voice": 0.3}
        scores = {}

        for k, v in results.items():
            if v not in scores:
                scores[v] = 0
            scores[v] += weights.get(k, 0)

        final = max(scores, key=scores.get)
        confidence = round(scores[final] * 100, 2)

        return jsonify({
            "face": results.get("face"),
            "text": results.get("text"),
            "voice": results.get("voice"),
            "final_emotion": final,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)