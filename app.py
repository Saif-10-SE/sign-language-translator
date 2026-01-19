"""
Streamlit app for real-time sign language translation using MediaPipe landmarks + trained model.
Run:
  streamlit run app.py
"""
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
import pyttsx3
from tensorflow.keras.models import load_model
from utils.landmark_utils import landmarks_to_list, normalize_landmarks
import os

st.set_page_config(page_title="Sign Language Translator", layout="wide")

# Sidebar controls
st.sidebar.title("Settings")
MODEL_PATH = st.sidebar.text_input("Model path", value="models/sign_model.h5")
LE_PATH = st.sidebar.text_input("Label encoder path", value="models/label_encoder.pkl")
CONF_THRESH = st.sidebar.slider("Prediction confidence threshold", 0.5, 0.99, 0.7, 0.01)
SPEAK = st.sidebar.checkbox("Enable speech (pyttsx3)", value=True)
ACCUMULATE_DELAY = st.sidebar.slider("Min seconds between accepted letters", 0.5, 3.0, 1.0, 0.1)

col1, col2 = st.columns([2,1])
with col1:
    st.header("Realtime Sign Language Translator")
    stframe = st.image([], channels="BGR")
with col2:
    st.header("Controls & Output")
    sentence_box = st.empty()
    st.write("Predicted letter + confidence:")
    pred_box = st.empty()
    clear_btn = st.button("Clear sentence")
    speak_btn = st.button("Speak sentence")

if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
    st.warning("Model or label encoder not found. Train the model first (see README).")
    st.stop()

model = load_model(MODEL_PATH)
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)

engine = None
if SPEAK:
    try:
        engine = pyttsx3.init()
    except Exception as e:
        st.warning(f"pyttsx3 init failed: {e}")
        engine = None

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
running = st.checkbox("Start camera", value=False)

# State across runs
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'last_time' not in st.session_state:
    st.session_state.last_time = 0.0
if 'last_letter' not in st.session_state:
    st.session_state.last_letter = ""


def speak_text(text):
    if engine:
        engine.say(text)
        engine.runAndWait()

try:
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as hands:
        while running:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to read from webcam.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            lm = landmarks_to_list(results, frame.shape)
            display_frame = frame.copy()
            pred_text = "-"
            conf = 0.0
            if lm is not None:
                norm = normalize_landmarks(lm)
                X = np.array(norm).reshape(1, -1).astype(np.float32)
                probs = model.predict(X, verbose=0)[0]
                idx = np.argmax(probs)
                conf = float(probs[idx])
                label = le.inverse_transform([idx])[0]
                pred_text = f"{label} ({conf:.2f})"
                # Accept letter if confidence is above threshold and enough time since last accept
                now = time.time()
                if conf >= CONF_THRESH and (now - st.session_state.last_time) >= ACCUMULATE_DELAY:
                    st.session_state.sentence += label
                    st.session_state.last_time = now
                    st.session_state.last_letter = label

                # draw the predicted label on the frame
                cv2.putText(display_frame, f"Pred: {label} ({conf:.2f})", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if conf>=CONF_THRESH else (0,0,255), 2)
            else:
                cv2.putText(display_frame, "No hand detected", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            stframe.image(display_frame, channels="BGR")
            sentence_box.markdown(f"**Sentence:** {st.session_state.sentence}")
            pred_box.markdown(f"**Last accepted letter:** {st.session_state.last_letter}  \n**Latest prediction:** {pred_text}")

            if clear_btn:
                st.session_state.sentence = ""
                st.session_state.last_letter = ""
                clear_btn = False

            if speak_btn:
                if st.session_state.sentence.strip():
                    speak_text(st.session_state.sentence)
                else:
                    speak_text("No text to speak")
                speak_btn = False

            # allow Streamlit to update UI
            time.sleep(0.03)
            # re-evaluate running checkbox state
            running = st.checkbox("Start camera", value=True)
finally:
    cap.release()
    cv2.destroyAllWindows()
