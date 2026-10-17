import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(
        self,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

    def extract_landmarks(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.hands.process(rgb)

        landmarks = []
        handedness = []
        out = frame.copy()

        if not results.multi_hand_landmarks:
            return out, landmarks, handedness

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            label = hand_info.classification[0].label
            handedness.append(label)

            pts = np.array(
                [[lm.x * w, lm.y * h, lm.z] for lm in hand_lm.landmark],
                dtype=np.float32,
            )
            landmarks.append(pts)

            if draw:
                self.mp_drawing.draw_landmarks(
                    out,
                    hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        return out, landmarks, handedness

    def close(self):
        self.hands.close()
