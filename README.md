Ce README est en français. Pour la version anglaise, voir [README_en.md](README_en.md).

# NewsVector — Système Intelligent de Classification d’Articles

## Description

**NewsVector** est un système NLP complet permettant de classifier automatiquement des articles d’actualité en quatre catégories : **World, Sports, Business et Sci/Tech**.  
Le projet repose sur des techniques modernes de traitement du langage naturel, combinant **embeddings sémantiques**, **Machine Learning**, **stockage vectoriel** et **orchestration automatisée**.

---

## Objectifs du projet

- Classifier automatiquement des articles d’actualité
- Utiliser des embeddings sémantiques multilingues
- Stocker les vecteurs dans une base vectorielle (ChromaDB)
- Entraîner et évaluer des modèles de Machine Learning
- Déployer une application interactive avec Streamlit
- Orchestrer le pipeline avec Apache Airflow

---

## Structure du projet

```
NewsVector/
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_embeddings.ipynb
│   ├── 05_chromadb_storage.ipynb
│   ├── 06_training.ipynb
│   └── 07_evaluation.ipynb
│
├── data/
│   ├── raw/                    # Données brutes
│   ├── processed/              # Données nettoyées
│   ├── embeddings/             # Vecteurs d’embeddings
│   ├── chromaDB/               # Collections stockée dans ChromaDB
│   └── metadata/               # Metadatas: Labels et id de chaque ligne
│
├── models/                     # Modèles ML
│
├── app/
│   └── streamlit_app.py        # Application Streamlit
│
├── airflow/
│   └── dags/
│       └── newsvector_dag.py
│
├── requirements.txt
├── README_en.md                # Version anglaise de README
├── README.md
├── Dockerfile
├── docker-compose.yaml         # Contient les services d'Airflow
├── .env
└── .gitignore
```

---

## Pipeline NLP

1. Chargement du dataset **SetFit/ag_news** depuis Hugging Face  
2. Analyse exploratoire des données (EDA)  
3. Prétraitement des textes :
    - Normalisation
    - Suppression des doublons
    - Suppression des stopwords
    - Suppression de la ponctuation (regex)
4. Génération des embeddings avec **Sentence Transformers**
5. Stockage des embeddings dans **ChromaDB**
6. Entraînement des modèles de Machine Learning
7. Évaluation des performances et détection de l’overfitting
8. Déploiement via **Streamlit**
9. Orchestration complète avec **Apache Airflow**

---

## Modèles et technologies utilisées

- **NLP & Embeddings**
    - Sentence Transformers
    - Modèle : `paraphrase-multilingual-MiniLM-L12-v2`

- **Machine Learning**
    - Scikit-learn (Logistic Regression, SVC, RandomForestClassifier, KNNClassifier)
    - XGBoost

- **Stockage vectoriel**
    - ChromaDB

- **Orchestration**
    - Apache Airflow

- Conteneurisation & Déploiement
    - Docker (pour apache airflow)

- **Interface utilisateur**
    - Streamlit

---

## Installation

```bash
git clone https://github.com/anass17/NewsVector
cd NewsVector
pip install -r requirements.txt
```

---

## Exécution du projet

1. Lancer les notebooks

Exécuter les notebooks dans l’ordre: 01 → 07

2. Lancer l’application Streamlit

```Bash
streamlit run app/streamlit_app.py
```

3. Lancer Airflow (avec Docker)

- Créer un fichier `.env` à la racine du projet :

```
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin
```

- Construire l’image et lancer les services Airflow :

```Bash
docker compose up --build
```

- Accéder à l’interface web d’Airflow :

URL : `http://localhost:8080`

- Identifiants :

Username : **admin**
Password : **admin**

---

## Visualisations

### Interface Streamlit

![Streamlit UI](https://github.com/user-attachments/assets/143b969d-1beb-4e9e-adbb-fbcef7ec3f17)
