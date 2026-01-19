"""
Capture hand landmarks with MediaPipe and save labeled samples to CSV.

Usage:
  python data_collect.py --label A --count 200

Each row in `data/landmarks.csv` will be:
label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
"""
import cv2
import mediapipe as mp
import argparse
import csv
import os
import time
from utils.landmark_utils import landmarks_to_list, normalize_landmarks

mp_hands = mp.solutions.hands


def ensure_data_dir():
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "landmarks.csv")
    if not os.path.exists(path):
        # write header
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"{c}{i}" for i in range(21) for c in ("x","y","z")]
            writer.writerow(header)
    return path


def collect(label: str, count: int, delay: float):
    csv_path = ensure_data_dir()
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as hands:
        saved = 0
        last_save = 0
        print(f"Starting collection for label '{label}' (target {count})")
        while saved < count:
            ret, frame = cap.read()
            if not ret:
                print("Can't access webcam.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            lm = landmarks_to_list(results, frame.shape)
            overlay = frame.copy()
            if lm is not None:
                norm = normalize_landmarks(lm)
                now = time.time()
                if now - last_save >= delay:
                    # save sample
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([label] + norm)
                    saved += 1
                    last_save = now
                    print(f"Saved {saved}/{count}")
                # draw text
                cv2.putText(overlay, f"Detected hand - saved {saved}/{count}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(overlay, f"No hand detected", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            cv2.imshow("Collecting landmarks - Press 'q' to quit", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Label for this gesture (e.g., A)")
    parser.add_argument("--count", type=int, default=200, help="Number of samples to collect")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds between saved samples")
    args = parser.parse_args()
    collect(args.label, args.count, args.delay)
