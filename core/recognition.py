import json
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import (
    COOLDOWN_FRAMES,
    LABELS_PATH,
    MODEL_PATH,
    PREDICTION_THRESHOLD,
    SEQUENCE_LENGTH,
    SMOOTHING_WINDOW,
)

from core.camera import HandDetector
from core.preprocessing import create_fixed_sequence, normalize_landmarks
from core.speech import SpeechService


class GestureRecognizer:
    def __init__(self, enable_voice=True):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. Train the model first."
            )

        if not LABELS_PATH.exists():
            raise FileNotFoundError(
                f"Labels file not found: {LABELS_PATH}. Prepare the dataset first."
            )

        self.model = load_model(MODEL_PATH)

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        self.detector = HandDetector()
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

        self.last_prediction = None
        self.last_confidence = 0.0
        self.cooldown_counter = 0

        self.speech_service = SpeechService() if enable_voice else None

    def set_language(self, language):
        if self.speech_service is not None:
            self.speech_service.set_language(language)

    def process_frame(self, frame, show_landmarks=True, enable_voice=False):
        out, landmarks, handedness = self.detector.extract_landmarks(
            frame,
            draw=show_landmarks,
        )

        if landmarks:
            normalized = normalize_landmarks(landmarks, handedness)
            self.sequence_buffer.append(normalized)

        if len(self.sequence_buffer) >= 5 and self.cooldown_counter <= 0:
            seq = create_fixed_sequence(self.sequence_buffer)
            seq = np.expand_dims(seq, axis=0)

            pred = self.model.predict(seq, verbose=0)[0]
            self.prediction_buffer.append(pred)

            if len(self.prediction_buffer) == SMOOTHING_WINDOW:
                avg = np.mean(self.prediction_buffer, axis=0)
                confidence = float(np.max(avg))
                idx = int(np.argmax(avg))

                if confidence >= PREDICTION_THRESHOLD:
                    result = self.labels[str(idx)]

                    self.last_prediction = result
                    self.last_confidence = confidence
                    self.cooldown_counter = COOLDOWN_FRAMES

                    if enable_voice and self.speech_service is not None:
                        tts_text = result.get("tts_text", result["display_text"])
                        self.speech_service.speak(tts_text, label=result["name"])

        else:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)

        if self.last_prediction is not None:
            text = f"{self.last_prediction['display_text']} ({self.last_confidence:.2f})"
            cv2.putText(
                out,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 180, 0),
                2,
            )

        return out, self.last_prediction, self.last_confidence

    def close(self):
        self.detector.close()

        if self.speech_service is not None:
            self.speech_service.close()
