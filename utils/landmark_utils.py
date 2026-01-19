"""
Utility functions for extracting and normalizing MediaPipe hand landmarks.
"""
from typing import List, Optional
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


def landmarks_to_list(results: Optional[mp.solutions.hands.Hands], image_shape) -> Optional[List[float]]:
    """
    Convert MediaPipe landmarks to a flat list [x0,y0,z0, x1,y1,z1, ...] normalized to image size.
    If no hand is detected, returns None.
    """
    if results is None:
        return None
    if not results.multi_hand_landmarks:
        return None

    h, w = image_shape[:2]
    # Use the first detected hand (for single-hand ASL alphabet)
    hand_landmarks = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand_landmarks.landmark:
        # MediaPipe landmarks x,y are normalized to [0,1]; z is relative (can be used as-is)
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def normalize_landmarks(landmarks: List[float]) -> List[float]:
    """
    Normalize landmarks so that the wrist (landmark 0) is at the origin and scale by max distance.
    landmarks: flat list of length 63 (21 * 3)
    """
    arr = np.array(landmarks).reshape(-1, 3)
    # Translate so wrist (index 0) is at origin
    origin = arr[0].copy()
    arr[:, 0] -= origin[0]
    arr[:, 1] -= origin[1]
    arr[:, 2] -= origin[2]
    # Scale by max absolute value to reduce scale variance
    max_val = np.max(np.abs(arr))
    if max_val > 0:
        arr = arr / max_val
    return arr.flatten().tolist()
