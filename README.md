# Sign Language Translator (Real-time) — Streamlit + MediaPipe + TensorFlow

This repository will host a complete, runnable sign language translator project. The app uses your webcam to capture hand landmarks (MediaPipe), classifies them with a TensorFlow model, and provides a Streamlit UI for real-time translation.

What this project includes (to be added via PR):
- app.py — Streamlit real-time app
- data_collect.py — dataset collection (landmarks CSV)
- train.py — training script
- utils/landmark_utils.py — landmark extraction helpers
- sign_model.py — model architecture helper
- requirements.txt — dependencies

Quick start after files are added:
1. Create/activate a Python environment (optional if you already have deps).
2. Install dependencies:
   - pip install -r requirements.txt
3. Collect data:
   - python data_collect.py --label A --count 200
   - Repeat for more labels (B, C, …). Data saved to `data/landmarks.csv`.
4. Train the model:
   - python train.py --data_path data/landmarks.csv --model_path models/sign_model.h5
5. Run the app:
   - streamlit run app.py
   - Open the local URL in your browser and click “Start camera”.

Notes:
- Designed for local use with a webcam (Windows/macOS/Linux).
- If you already have Streamlit, OpenCV, MediaPipe, TensorFlow installed, you can skip `pip install -r requirements.txt`.
- Cloud platforms typically don’t support live webcam; local is recommended.

Next step:
- After committing this README, ask Copilot to “open a PR to add the project files” and the full project will be added here.
