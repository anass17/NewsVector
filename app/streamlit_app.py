import streamlit as st
import joblib as jb
from sentence_transformers import SentenceTransformer
import re
import pandas as pd

st.set_page_config(page_title="NewsVector", layout="wide", page_icon='📰')
st.title("NewsVector - Système intelligent de classification d'articles d'actualité")

running = True

try:
    svm_model = jb.load('./models/svm_model.pkl')
except:
    st.warning("Error! Could not load the SVM model")
    running = False
else:
    st.success("The SVM model was loaded successfully")

try:
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
except:
    st.warning("Error! Could not load the embedding model")
    running = False
else:
    st.success("The embedding model was loaded successfully")

classes = pd.read_csv('./data/processed/classes.csv')

if running == True:
    st.subheader('Prédire la classe d\'un article')
    article_text = st.text_area('Entrer un article:', height=250)

    if st.button('Prédire'):
        if article_text:

            article_text = article_text.strip()
            article_text = re.sub(r"\s+", " ", article_text)

            article_list = [article_text]

            train_embeddings = embedding_model.encode(
                article_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            prediction = svm_model.predict([train_embeddings])

            st.divider()

            st.subheader("Résultat:")

            st.info(f"L'article appartient à: ***{list(classes[classes['label'] == prediction[0]]['label_text'])[0]}***")


        else:
            st.error('Veuillez entrer un article!')