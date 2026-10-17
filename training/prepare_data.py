import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_DIR, LABELS_PATH, MODELS_DIR, REPORTS_DIR, SEQUENCE_LENGTH
from core.camera import HandDetector
from core.preprocessing import create_fixed_sequence, normalize_landmarks


SLIDING_STRIDE = 5


def create_labels(class_names):
    return {
        str(i): {
            "id": i,
            "name": name,
            "display_text": name,
            "tts_text": name,
        }
        for i, name in enumerate(class_names)
    }


def extract_frames(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        _, landmarks, handedness = detector.extract_landmarks(frame, draw=False)
        if landmarks:
            frames.append(normalize_landmarks(landmarks, handedness))

    cap.release()
    return frames


def create_sliding_sequences(frames):
    if len(frames) == 0:
        return []

    if len(frames) <= SEQUENCE_LENGTH:
        return [create_fixed_sequence(frames)]

    seqs = []
    for start in range(0, len(frames) - SEQUENCE_LENGTH + 1, SLIDING_STRIDE):
        seqs.append(create_fixed_sequence(frames[start:start + SEQUENCE_LENGTH]))

    if not seqs:
        seqs.append(create_fixed_sequence(frames))

    return seqs


def prepare_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    folders = sorted(f for f in dataset_path.iterdir() if f.is_dir())

    if len(folders) != 20:
        raise ValueError(f"The dataset must contain 20 folders. Found: {len(folders)}")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    class_names = [f.name for f in folders]
    labels = create_labels(class_names)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    detector = HandDetector()

    X, y, groups = [], [], []
    skipped = []
    group_id = 0

    for i, folder in enumerate(folders):
        videos = sorted(folder.glob("*.mp4"))
        print(f"Processing {folder.name}: {len(videos)} videos")

        for video_path in videos:
            seqs = create_sliding_sequences(extract_frames(video_path, detector))

            if not seqs:
                skipped.append(str(video_path))
                continue

            for seq in seqs:
                X.append(seq)
                y.append(i)
                groups.append(group_id)

            group_id += 1

    detector.close()

    if not X:
        raise ValueError("No valid sequences were extracted from the dataset.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups, dtype=np.int32)

    out = REPORTS_DIR / "dataset_sequences.npz"
    np.savez_compressed(out, X=X, y=y, groups=groups)

    report = {
        "dataset_path": str(dataset_path),
        "n_classes": len(class_names),
        "classes": class_names,
        "n_sequences": int(len(X)),
        "n_videos": int(group_id),
        "skipped_videos": skipped,
        "sliding_stride": SLIDING_STRIDE,
        "sequence_length": SEQUENCE_LENGTH,
    }

    with open(REPORTS_DIR / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print("Dataset preparation completed.")
    print(f"Sequences saved in: {out}")
    print(f"Labels saved in: {LABELS_PATH}")
    print(f"Total sequences: {len(X)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    args = parser.parse_args()
    prepare_dataset(args.dataset)


if __name__ == "__main__":
    main()
