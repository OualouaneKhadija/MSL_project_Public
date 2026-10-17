from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    LSTM,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam

from config import N_COORDINATES, N_LANDMARKS, SEQUENCE_LENGTH


def build_model(n_classes):
    model = Sequential(
        [
            TimeDistributed(
                Conv1D(64, kernel_size=3, activation="relu", padding="same"),
                input_shape=(SEQUENCE_LENGTH, N_LANDMARKS, N_COORDINATES),
            ),
            TimeDistributed(BatchNormalization()),

            TimeDistributed(
                Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            ),
            TimeDistributed(BatchNormalization()),

            TimeDistributed(Flatten()),

            LSTM(128, return_sequences=True),
            Dropout(0.35),

            LSTM(64),

            Dense(64, activation="relu"),
            Dropout(0.3),

            Dense(n_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
