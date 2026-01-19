"""
Train a classifier on collected landmarks CSV and save model + label encoder.

Usage:
  python train.py --data_path data/landmarks.csv --model_path models/sign_model.h5
"""
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sign_model import build_mlp
import tensorflow as tf


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['label'].astype(str).values
    X = df.drop(columns=['label']).values.astype(np.float32)
    return X, labels


def main(data_path, model_path, epochs, batch_size):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    X, y = load_data(data_path)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    num_classes = len(le.classes_)
    input_dim = X.shape[1]
    print(f"Data: {X.shape[0]} samples, input_dim={input_dim}, classes={num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

    model = build_mlp(input_dim, num_classes)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks)

    # Save label encoder
    with open(os.path.join(os.path.dirname(model_path), "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("Training complete. Model and label encoder saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/landmarks.csv")
    parser.add_argument("--model_path", default="models/sign_model.h5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args.data_path, args.model_path, args.epochs, args.batch_size)
