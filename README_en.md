This README is written in English. For the French version, see [README.md](README.md).

# NewsVector — Intelligent News Classification System

## Description

**NewsVector** is a complete NLP system designed to automatically classify news articles into four categories: **World, Sports, Business, and Sci/Tech**.  
The project relies on modern Natural Language Processing techniques, combining **semantic embeddings**, **Machine Learning**, **vector storage**, and **automated orchestration**.

---

## Project Objectives

- Automatically classify news articles
- Use multilingual semantic embeddings
- Store vectors in a vector database (ChromaDB)
- Train and evaluate Machine Learning models
- Deploy an interactive application using Streamlit
- Orchestrate the full pipeline with Apache Airflow

---

## Project Structure

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
│   ├── raw/                    # Raw data
│   ├── processed/              # Cleaned data
│   ├── embeddings/             # Embedding vectors
│   ├── chromaDB/               # Collections stored in ChromaDB
│   └── metadata/               # Metadata: labels and row IDs
│
├── models/                     # ML models
│
├── app/
│   └── streamlit_app.py        # Streamlit application
│
├── airflow/
│   └── dags/
│   └── newsvector_dag.py
│
├── requirements.txt
├── README_en.md # English version of the README
├── README.md
├── Dockerfile
├── docker-compose.yaml # Airflow services
├── .env
└── .gitignore
```

---

## NLP Pipeline

1. Load the **SetFit/ag_news** dataset from Hugging Face  
2. Perform Exploratory Data Analysis (EDA)  
3. Text preprocessing:
    - Text normalization
    - Duplicate removal
    - Stopwords removal
    - Punctuation removal (regex)
4. Generate embeddings using **Sentence Transformers**
5. Store embeddings in **ChromaDB**
6. Train Machine Learning models
7. Evaluate model performance and detect overfitting
8. Deploy the application using **Streamlit**
9. Fully orchestrate the pipeline with **Apache Airflow**

---

## Models and Technologies Used

- **NLP & Embeddings**
    - Sentence Transformers
    - Model: `paraphrase-multilingual-MiniLM-L12-v2`

- **Machine Learning**
    - Scikit-learn (Logistic Regression, SVC, RandomForestClassifier, KNNClassifier)
    - XGBoost

- **Vector Storage**
    - ChromaDB

- **Orchestration**
    - Apache Airflow

- **Containerization & Deployment**
    - Docker (used for Apache Airflow)

- **User Interface**
    - Streamlit

---

## Installation

```bash
git clone https://github.com/anass17/NewsVector
cd NewsVector
pip install -r requirements.txt
```

---

## Project Execution

1. Run the notebooks

Execute the notebooks in the following order: 01 → 07

2. Launch the Streamlit application

```Bash
streamlit run app/streamlit_app.py
```

3. Launch Airflow (with Docker)

- Create a `.env` file at the root of the project:

```
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin
```

- Build the image and start the Airflow services:

```Bash
docker compose up --build
```

- Access the Airflow web interface:

URL: `http://localhost:8080`

- Credentials:

Username: **admin**
Password: **admin**

---

## Visualizations

### Streamlit Interface

![Streamlit UI](https://github.com/user-attachments/assets/143b969d-1beb-4e9e-adbb-fbcef7ec3f17)
