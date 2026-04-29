import cv2
import numpy as np

def preprocess_face(face_img, size=(48,48)):
    
    if face_img.ndim == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face_img, size)
    face = face.reshape(1, size[0], size[1], 1).astype('float32') / 255.0
    return face
