# Sign Language Translator (Real-time) — Streamlit + MediaPipe + TensorFlow

This project captures hand landmarks with MediaPipe, trains a classifier on those landmarks, and runs a Streamlit app for real-time sign-to-text translation (ASL alphabet style). It is designed for local use.

Features:
- Dataset collection (landmarks only) with webcam
- Train an MLP classifier on landmark features
- Streamlit realtime translator: shows webcam, predicted letter, builds words, and speaks text

Requirements and quick start:
1. Clone or copy the project files locally.
2. Create a Python virtual environment (recommended):
   - python -m venv venv
   - source venv/bin/activate  (macOS/Linux) or venv\Scripts\activate (Windows)
3. Install dependencies:
   - pip install -r requirements.txt
4. Collect data:
   - python data_collect.py --label A --count 200
   - Repeat for each label you want (A, B, C, ...). Samples are appended to `data/landmarks.csv`.
5. Train model:
   - python train.py --data_path data/landmarks.csv --model_path models/sign_model.h5
   - This creates `models/sign_model.h5` and `models/label_encoder.pkl`.
6. Run the Streamlit app:
   - streamlit run app.py
   - Open the provided local URL.

Notes and tips:
- The simplest dataset: collect static ASL alphabet letters. For better performance, collect at least several hundred samples per class from multiple angles/lighting.
- The model uses 21 hand landmarks (x, y, z) flattened to a 63-length vector.
- If you prefer images, extend the dataset and model to use CNNs; this project focuses on landmarks (lightweight + robust).
- If you encounter webcam permission problems on Streamlit, run the script in a terminal and ensure OpenCV can access the camera.

Project layout (files provided):
- app.py — Streamlit real-time app
- data_collect.py — dataset collection (landmarks CSV)
- train.py — training script
- utils/landmark_utils.py — landmark extraction helpers
- sign_model.py — model architecture helper
- requirements.txt
- README.md

If you want, I can:
- Add pre-made dataset loader and example dataset
- Replace the MLP with an LSTM (for dynamic gestures) or a CNN (image-based)
- Provide a Dockerfile for easy deployment
- Create a gif/video demo and test suite
