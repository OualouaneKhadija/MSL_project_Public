import json
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import HISTORY_PATH, MODEL_PATH, REPORTS_DIR
from core.model import build_model


N_AUGMENTATIONS = 3


def split_by_video_groups(X, y, groups, test_size=0.2):
    train_groups, test_groups = [], []
    rng = np.random.default_rng(42)

    for class_id in np.unique(y):
        class_groups = np.unique(groups[y == class_id])
        rng.shuffle(class_groups)

        n_test = max(1, int(round(len(class_groups) * test_size)))
        test_groups.extend(class_groups[:n_test])
        train_groups.extend(class_groups[n_test:])

    train_mask = np.isin(groups, train_groups)
    test_mask  = np.isin(groups, test_groups)

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def augment_sequence(seq):
    aug = seq.copy()
    aug += np.random.normal(0, 0.015, aug.shape)
    aug *= np.random.uniform(0.95, 1.05)
    aug += np.random.uniform(-0.03, 0.03, size=(1, 1, 3))
    return aug.astype(np.float32)


def augment_training_data(X_train, y_train):
    X_aug, y_aug = [], []

    for seq, label in zip(X_train, y_train):
        X_aug.append(seq)
        y_aug.append(label)
        for _ in range(N_AUGMENTATIONS):
            X_aug.append(augment_sequence(seq))
            y_aug.append(label)

    return (
        np.asarray(X_aug, dtype=np.float32),
        np.asarray(y_aug, dtype=np.int32),
    )


def calculate_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n = len(classes)
    return {int(c): float(total / (n * cnt)) for c, cnt in zip(classes, counts)}


def train_model(epochs=35, batch_size=8):
    dataset_file = REPORTS_DIR / "dataset_sequences.npz"

    if not dataset_file.exists():
        raise FileNotFoundError(
            "Prepared dataset not found. Run training/prepare_data.py first."
        )

    data = np.load(dataset_file)
    X, y, groups = data["X"], data["y"], data["groups"]

    n_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = split_by_video_groups(X, y, groups)
    X_train, y_train = augment_training_data(X_train, y_train)

    class_weights = calculate_class_weights(y_train)

    print(f"Train samples after augmentation: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")

    model = build_model(n_classes)

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=4)

    model.save(MODEL_PATH)

    print("Training completed.")
    print(f"Model saved in: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
