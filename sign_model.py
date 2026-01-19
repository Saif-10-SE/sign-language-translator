"""
Model architecture helper for the sign language classifier.
Simple MLP that takes 63-dim landmark input and predicts class probabilities.
"""
from tensorflow.keras import layers, models


def build_mlp(input_dim: int, num_classes: int):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
