
import numpy as np
from tensorflow.keras.models import load_model
import cv2

CLASS_MAP = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

class EmotionPredictor:
    def __init__(self, model_path):
        print('Loading model from', model_path)
        self.model = load_model(model_path)

    def predict(self, face_img):
        
        if face_img.ndim == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (48,48))
        x = face_img.reshape(1,48,48,1).astype('float32')/255.0
        probs = self.model.predict(x)[0]
        idx = int(np.argmax(probs))
        return {'label': CLASS_MAP[idx], 'confidence': float(probs[idx])}
