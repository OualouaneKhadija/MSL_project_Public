import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

HISTORY_PATH = REPORTS_DIR / "training_history.json"
EVALUATION_PATH = REPORTS_DIR / "evaluation.json"

MODEL_NAME = "Conv1D-LSTM"

TRAIN_COLOR = "#1F77B4"
VALIDATION_COLOR = "#FF7F0E"


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})


def load_json_file(path, filename):
    if not path.exists():
        print(f"Fichier introuvable : {filename}")
        print(f"Chemin attendu : {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure sauvegardée : {output_path.name}")


def plot_accuracy(history):
    train_accuracy = history.get("accuracy", history.get("acc", []))
    validation_accuracy = history.get("val_accuracy", history.get("val_acc", []))

    if not train_accuracy or not validation_accuracy:
        print("Courbe de précision ignorée : données manquantes.")
        return

    epochs = range(1, len(train_accuracy) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        train_accuracy,
        color=TRAIN_COLOR,
        linewidth=1.8,
        label="Entraînement",
    )

    ax.plot(
        epochs,
        validation_accuracy,
        color=VALIDATION_COLOR,
        linewidth=1.8,
        label="Validation",
    )

    ax.set_title(f"Précision du modèle {MODEL_NAME}", fontsize=13, pad=10)
    ax.set_xlabel("Époque")
    ax.set_ylabel("Précision")
    ax.set_xlim(1, len(train_accuracy))
    ax.set_ylim(0, 1.02)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, f"accuracy_{MODEL_NAME}.png")


def plot_loss(history):
    train_loss = history.get("loss", [])
    validation_loss = history.get("val_loss", [])

    if not train_loss or not validation_loss:
        print("Courbe de perte ignorée : données manquantes.")
        return

    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        train_loss,
        color=TRAIN_COLOR,
        linewidth=1.8,
        label="Entraînement",
    )

    ax.plot(
        epochs,
        validation_loss,
        color=VALIDATION_COLOR,
        linewidth=1.8,
        label="Validation",
    )

    ax.set_title(f"Perte du modèle {MODEL_NAME}", fontsize=13, pad=10)
    ax.set_xlabel("Époque")
    ax.set_ylabel("Perte")
    ax.set_xlim(1, len(train_loss))
    ax.legend()

    fig.tight_layout()
    save_figure(fig, f"loss_{MODEL_NAME}.png")


def plot_f1_scores(evaluation):
    class_report = evaluation.get("class_report", {})

    if not class_report:
        print("Graphique F1-score ignoré : rapport par classe manquant.")
        return

    class_names = list(class_report.keys())
    f1_scores = [class_report[name]["f1_score"] for name in class_names]

    fig_width = max(10, len(class_names) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    bars = ax.bar(
        class_names,
        f1_scores,
        color=TRAIN_COLOR,
        width=0.6,
        edgecolor="white",
        zorder=3,
    )

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(f"F1-score par classe ({MODEL_NAME})", fontsize=13, pad=10)
    ax.set_xlabel("Classe")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)

    fig.tight_layout()
    save_figure(fig, f"f1_score_{MODEL_NAME}.png")


def plot_confusion_matrix(evaluation):
    matrix = evaluation.get("confusion_matrix")
    class_report = evaluation.get("class_report", {})

    if matrix is None:
        print("Matrice de confusion ignorée : données manquantes.")
        return

    matrix = np.array(matrix)
    number_of_classes = len(matrix)

    if class_report:
        labels = list(class_report.keys())
    else:
        labels = [str(index) for index in range(number_of_classes)]

    fig_size = max(8, number_of_classes * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    image = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(number_of_classes))
    ax.set_yticks(range(number_of_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0

    for i in range(number_of_classes):
        for j in range(number_of_classes):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    ax.set_title(f"Matrice de confusion ({MODEL_NAME})", fontsize=13, pad=10)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")

    fig.tight_layout()
    save_figure(fig, f"confusion_matrix_{MODEL_NAME}.png")


def plot_precision_recall(evaluation):
    class_report = evaluation.get("class_report", {})

    if not class_report:
        print("Graphique précision/rappel ignoré : rapport par classe manquant.")
        return

    class_names = list(class_report.keys())
    precision_scores = [class_report[name]["precision"] for name in class_names]
    recall_scores = [class_report[name]["recall"] for name in class_names]

    x_positions = np.arange(len(class_names))
    bar_width = 0.35

    fig_width = max(10, len(class_names) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    ax.bar(
        x_positions - bar_width / 2,
        precision_scores,
        bar_width,
        label="Précision",
        color=TRAIN_COLOR,
        edgecolor="white",
        zorder=3,
    )

    ax.bar(
        x_positions + bar_width / 2,
        recall_scores,
        bar_width,
        label="Rappel",
        color=VALIDATION_COLOR,
        edgecolor="white",
        zorder=3,
    )

    ax.set_title(f"Précision et rappel par classe ({MODEL_NAME})", fontsize=13, pad=10)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, f"precision_recall_{MODEL_NAME}.png")


def main():
    if not HISTORY_PATH.exists():
        print("Le fichier training_history.json est introuvable.")
        print("Lance d'abord : python training/train.py")
        sys.exit(1)

    if not EVALUATION_PATH.exists():
        print("Le fichier evaluation.json est introuvable.")
        print("Lance d'abord : python training/evaluate.py")
        sys.exit(1)

    history = load_json_file(HISTORY_PATH, "training_history.json")
    evaluation = load_json_file(EVALUATION_PATH, "evaluation.json")

    accuracy = evaluation.get("accuracy", 0)
    epochs_count = len(history.get("loss", []))
    test_sequences = evaluation.get(
        "number_of_test_sequences",
        evaluation.get("n_test_sequences", "?"),
    )

    print("Génération des courbes d'évaluation")
    print(f"Modèle : {MODEL_NAME}")
    print(f"Époques : {epochs_count}")
    print(f"Accuracy test : {accuracy:.4f}")
    print(f"Séquences de test : {test_sequences}")
    print(f"Dossier de sortie : {FIGURES_DIR}")
    print()

    plot_accuracy(history)
    plot_loss(history)
    plot_f1_scores(evaluation)
    plot_confusion_matrix(evaluation)
    plot_precision_recall(evaluation)

    print()
    print("Génération terminée.")


if __name__ == "__main__":
    main()
