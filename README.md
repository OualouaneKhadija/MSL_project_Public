# MSL Project : Reconnaissance de la langue des signes marocaine

Application web de reconnaissance de signes à partir de vidéos, basée sur l'apprentissage profond.
Le projet couvre toute la chaîne : préparation des données, entraînement, évaluation et démonstration via une application Flask.

> **Note** : le dataset (vidéos) n'est volontairement **pas inclus** dans ce dépôt. Voir la section [Données](#données).

## Fonctionnalités

- Reconnaissance de **20 signes** (classes) de la langue des signes marocaine
- Extraction de séquences à partir de vidéos
- Entraînement et évaluation d'un modèle de classification de séquences
- Génération de rapports et de courbes d'apprentissage
- Interface web (Flask) pour tester le modèle

## Structure du projet

```
MSL_project/
├── app.py              # Application Flask (point d'entrée)
├── config.py           # Configuration (chemins, hyperparamètres)
├── plot_courbes.py     # Tracé des courbes d'entraînement
├── requirements.txt    # Dépendances Python
├── core/               # Logique principale (traitement, prédiction)
├── models/             # Modèles entraînés
├── training/           # Scripts : prepare_data.py, train.py, evaluate.py
├── reports/            # Rapports d'évaluation et figures
├── static/             # Fichiers statiques (CSS, JS)
└── templates/          # Pages HTML
```

## Installation

```bash
git clone https://github.com/OualouaneKhadija/MSL_project.git
cd MSL_project

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

## Données

Le dataset n'est pas fourni. Pour utiliser le projet avec vos propres vidéos, organisez-les ainsi (un dossier par classe) :

```
dataset/
├── classe_1/
│   ├── video_01.mp4
│   └── video_02.mp4
├── classe_2/
│   └── ...
└── ...
```

Indiquez ensuite l'emplacement du dataset, soit en modifiant `config.py`, soit avec une variable d'environnement :

```bash
set MSL_DATASET_PATH=chemin\vers\dataset      # Windows
# export MSL_DATASET_PATH=chemin/vers/dataset # Linux / macOS
```

## Utilisation

**1. Préparer les données**
```bash
python training/prepare_data.py
```

**2. Entraîner le modèle**
```bash
python training/train.py
```

**3. Évaluer le modèle**
```bash
python training/evaluate.py
```

**4. Tracer les courbes d'apprentissage**
```bash
python plot_courbes.py
```

**5. Lancer l'application web**
```bash
python app.py
```
Puis ouvrir http://127.0.0.1:5000 dans le navigateur.

## Résultats

Évaluation sur un jeu de test de 488 séquences, 20 classes :

| Métrique | Valeur |
|----------|--------|
| Accuracy (test) | ≈ 61,7 % |

Les métriques détaillées par classe (précision, rappel, F1) sont disponibles dans `reports/evaluation.json`, et l'historique d'entraînement dans `reports/training_history.json`.

## Limites et pistes d'amélioration

- Dataset de taille limitée : l'écart entre la performance d'entraînement et de test indique un risque de surapprentissage.
- Pistes : plus de données et de variété (signeurs, éclairages, angles), augmentation de données, régularisation, tests sur d'autres architectures.

## Technologies

- Python
- Flask
- NumPy
- Framework de deep learning (voir `requirements.txt`)

## Auteure

**Khadija Oualouane**
GitHub : [@OualouaneKhadija](https://github.com/OualouaneKhadija)
