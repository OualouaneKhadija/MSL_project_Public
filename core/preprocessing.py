import numpy as np

from config import N_COORDINATES, N_LANDMARKS, SEQUENCE_LENGTH


def normalize_landmarks(landmarks, handedness):
    arr = np.zeros((N_LANDMARKS, N_COORDINATES), dtype=np.float32)

    if not landmarks:
        return arr

    sorted_hands = sorted(
        zip(handedness, landmarks),
        key=lambda x: 0 if x[0] == "Right" else 1,
    )

    for i, (_, pts) in enumerate(sorted_hands[:2]):
        wrist = pts[0]
        centered = pts - wrist

        scale = np.max(np.linalg.norm(centered, axis=1))
        if scale > 1e-7:
            centered = centered / scale

        start = i * 21
        arr[start:start + 21] = centered[:, :3]

    return arr


def create_fixed_sequence(frames, sequence_length=SEQUENCE_LENGTH):
    frames = np.asarray(list(frames), dtype=np.float32)

    if len(frames) == 0:
        return np.zeros(
            (sequence_length, N_LANDMARKS, N_COORDINATES),
            dtype=np.float32,
        )

    if len(frames) == sequence_length:
        return frames

    if len(frames) < sequence_length:
        padding = np.repeat(frames[-1:], sequence_length - len(frames), axis=0)
        return np.concatenate([frames, padding], axis=0)

    indices = np.linspace(0, len(frames) - 1, sequence_length).astype(int)
    return frames[indices]
