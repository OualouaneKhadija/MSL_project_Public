import json
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import EVALUATION_PATH, LABELS_PATH, MODEL_PATH, REPORTS_DIR


def split_test_data_by_groups(X, y, groups, test_size=0.2):
    test_groups = []
    rng = np.random.default_rng(42)

    for class_id in np.unique(y):
        class_groups = np.unique(groups[y == class_id])
        rng.shuffle(class_groups)

        n_test = max(1, int(round(len(class_groups) * test_size)))
        test_groups.extend(class_groups[:n_test])

    mask = np.isin(groups, test_groups)
    return X[mask], y[mask]


def create_confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1

    return matrix.tolist()


def create_class_report(y_true, y_pred, class_names):
    report = {}

    for i, name in enumerate(class_names):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[name] = {"precision": precision, "recall": recall, "f1_score": f1}

    return report


def evaluate_model():
    dataset_file = REPORTS_DIR / "dataset_sequences.npz"

    if not dataset_file.exists():
        raise FileNotFoundError(
            "Prepared dataset not found. Run training/prepare_data.py first."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run training/train.py first.")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    class_names = [labels[str(i)]["name"] for i in range(len(labels))]

    data = np.load(dataset_file)
    X, y, groups = data["X"], data["y"], data["groups"]

    X_test, y_test = split_test_data_by_groups(X, y, groups)

    model = load_model(MODEL_PATH)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    accuracy = float(np.mean(y_test == y_pred))

    report = {
        "accuracy": accuracy,
        "n_test_sequences": int(len(X_test)),
        "class_report": create_class_report(y_test, y_pred, class_names),
        "confusion_matrix": create_confusion_matrix(y_test, y_pred, len(class_names)),
    }

    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print("Evaluation completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Test sequences: {len(X_test)}")
    print(f"Report saved in: {EVALUATION_PATH}")


if __name__ == "__main__":
    evaluate_model()
