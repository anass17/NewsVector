from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from sklearn.svm import SVC
import joblib

DATA_PATH = "/opt/airflow/data"
MODEL_PATH = "/opt/airflow/models"
CHROMADB_PATH = "/opt/airflow/chroma_db"
METADATA_PATH = "/opt/airflow/metadatas"


# Function to load the dataset and save as CSV files
def load_data() :
    dataset = load_dataset("SetFit/ag_news")
    
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    df_train = df_train.iloc[:5000]
    df_test = df_test.iloc[:5000]
    
    df_train.to_csv(os.path.join(DATA_PATH, "train.csv"), index=False)
    df_test.to_csv(os.path.join(DATA_PATH, "test.csv"), index=False)

    print("Datasets Enregistrées avec Succès !")
    


# Function to generate embeddings using SentenceTransformer
def generate_embeddings() :
    df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    df_train["text"] = df_train["text"].fillna("").str.strip()
    df_train["text"] = df_train["text"].str.replace(r"\s+", " ", regex=True)

    df_test["text"] = df_test["text"].fillna("").str.strip()
    df_test["text"] = df_test["text"].str.replace(r"\s+", " ", regex=True)

    texts_train = df_train["text"].tolist()
    texts_test = df_test["text"].tolist()
    
    embeddings_train = model.encode(texts_train, normalize_embeddings=True, show_progress_bar=True)
    embeddings_test = model.encode(texts_test, normalize_embeddings=True, show_progress_bar=True)
    
    df_train["text_embedding"] = embeddings_train.tolist()
    df_test["text_embedding"] = embeddings_test.tolist()
    
    df_train["id"] = "train_art_" + df_train.index.astype(str)
    df_test["id"] = "test_art_" + df_test.index.astype(str)

    df_train.to_pickle(os.path.join(METADATA_PATH, "train.pkl"))
    df_test.to_pickle(os.path.join(METADATA_PATH, "test.pkl"))

    print("Metadatas Sauvegardés avec Succès !")



# Function to save embeddings in ChromaDB
def save_embeddings_chromadb() :
    
    df_train = pd.read_pickle(os.path.join(METADATA_PATH, "train.pkl"))
    df_test = pd.read_pickle(os.path.join(METADATA_PATH, "test.pkl"))
    
    client = chromadb.PersistentClient(path=CHROMADB_PATH)

    collection_train = client.create_collection(
        name="train_news",
        metadata={"description": "Embeddings pour données d'entraînement"}
    )
    collection_test = client.create_collection(
        name="test_news",
        metadata={"description": "Embeddings pour données de test"}
    )
    
    batch_size = 5000

    dfs = [df_train, df_test]
    collections = [collection_train,collection_test]
        
    for index, df in enumerate(dfs) :
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]

            batch_ids = batch["id"].astype(str).tolist()
            batch_embeddings = batch["text_embedding"].tolist()
            batch_metadata = batch[["label", "text"]].to_dict("records")

            collections[index].add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadata
            )

        if index == 0 :
            print(f"Total Items Train : {collection_test.count()}")
        else :
            print(f"Total Items Test : {collection_train.count()}")
    
    print("Embeddings inserées dans ChromaDB...")
            


# Function to train the model MLPClassifier
def train_model() :
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collections = client.list_collections()

    for col in collections:
        print(f"- {col.name}")
        
    train_collection = client.get_collection(name=collections[1].name)
    test_collection = client.get_collection(name=collections[0].name)
    
    train_data = train_collection.get(include=['embeddings','metadatas'])
    test_data = test_collection.get(include=['embeddings','metadatas'])
    
    X_train = np.array(train_data['embeddings'])
    X_test = np.array(test_data['embeddings'])

    y_train = np.array([data['label'] for data in train_data['metadatas']])
    y_test = np.array([data['label'] for data in test_data['metadatas']])

    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    svm_model.fit(X_train, y_train)
    
    joblib.dump(svm_model, os.path.join(MODEL_PATH, "model.pkl"))

    print("Modèle Sauvegardé avec Succès !")



DEFAULT_ARGS = {
    "owner": "news-classifier-pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 12, 12),
}


with DAG(
    dag_id="news_classifier_pipeline_dag",
    default_args=DEFAULT_ARGS,
    description="DAG pour le pipeline de classification des news",
    schedule_interval="0 17 * * 1",
    catchup=False,
) as dag:
    
    t_load_data = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )
    
    t_generate_embeddings = PythonOperator(
        task_id="generate_embeddings",
        python_callable=generate_embeddings
    )
    
    t_save_embeddings_chromadb = PythonOperator(
        task_id="save_embeddings_chromadb",
        python_callable=save_embeddings_chromadb
    )
    
    t_train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )
    
    t_load_data >> t_generate_embeddings >> t_save_embeddings_chromadb >> t_train_model