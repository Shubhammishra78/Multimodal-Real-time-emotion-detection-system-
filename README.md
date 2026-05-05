# Realtime Face Emotion Detection - Full Project

This project contains training scripts, a Flask realtime server, and a browser-based frontend for webcam emotion detection.

## Folder structure

- training/: Training scripts (train.py)
- app/: Flask + Socket.IO server and inference wrapper
- web/: Frontend (index.html, main.js, styles.css)
- models/: Save trained Keras model here as `emotion_model.h5`
- data/: Place `fer2013.csv` here to train
- presentation/: Slides and demo script

## Quick steps to run on Windows (VS Code)

1. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Train model:
   - Download FER2013 from Kaggle and put `fer2013.csv` into `data/`.
   - Run training:
     ```bash
     cd training
     python train.py --csv ../data/fer2013.csv --out ../models/emotion_model.h5
     ```
   Training on CPU may be slow. Use Google Colab for GPU training if needed.

4. Run server:
   ```bash
   cd app
   python app.py
   ```
   Open browser: http://localhost:5000/index.html

5. If the model is missing, the server will tell you; add `models/emotion_model.h5`.

## If you prefer a pretrained model
I can provide a pretrained `emotion_model.h5` (about tens of MB). Say "Include pretrained model" and I'll add it to the ZIP.

## Notes
- Use Chrome or Edge for webcam access.
- If TensorFlow installation fails, try `pip install tensorflow-cpu` or use Colab.
