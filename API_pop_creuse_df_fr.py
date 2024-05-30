#!/usr/bin/env python
# coding: utf-8

# In[1]:

pip install scikit-learn

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import requests


# In[2]:


# Load the dataset
df_fr = pd.read_csv(r"C:\Users\Alan\Downloads\df_fr_final2405.csv")


# In[ ]:


# Streamlit setup
st.set_page_config(
    page_title="Pop Creuse",
    layout="centered",
    page_icon=':popcorn:'
)

# Background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/free-photo/cinema-still-life_23-2148017314.jpg");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# In[ ]:


# Dropdown options
list_film_deroulante_films = ["Tape le film que tu aimes"] + list(df_fr["originalTitle"])
list_personne_deroulante = ["Tape la personne dont tu veux voir les films"] + list(df_fr.columns[33:])


# In[ ]:


# API settings
url_api = "http://www.omdbapi.com/?i="
key_api = "&apikey=b0402387"
url_imdb = 'https://www.imdb.com/title/'


# In[ ]:


# Define feature weights
array_size = 5519  # Ensure this matches the number of features
feature_weights = np.zeros(array_size, dtype=int)
feature_weights[0:5] = 1
feature_weights[5] = 3
feature_weights[6] = 1
feature_weights[7:31] = 10
feature_weights[31:] = 2

# Extract only the relevant features for training
X = df_fr.iloc[:, 2:]  # Assuming first two columns are 'originalTitle' and 'tconst'

# Check if feature weights match the number of features
if len(feature_weights) != X.shape[1]:
    st.error("The size of feature_weights does not match the number of features in X.")
    st.stop()

# Step 1: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply the feature weights
X_weighted = X_scaled * feature_weights

# Step 3: Use NearestNeighbors with the transformed data
nn = NearestNeighbors(n_neighbors=4)
nn.fit(X_weighted)


# In[ ]:


with st.form("form 1"):
    st.subheader("OPTION 1 : Choisi ton acteur ou réalisateur et selectionne 'Demander'")
    personne = st.selectbox("Personne : ", list_personne_deroulante)
    submit_1 = st.form_submit_button("Demander")

if submit_1:
    list_film_deroulante_films = list(df_fr["originalTitle"][df_fr[personne] == 1])
    if list_film_deroulante_films.empty:
        list_film_deroulante_films = list(df_fr["originalTitle"])

with st.form("form 2"):
    st.subheader("OPTION 2 : Choisi ton film préféré ♥️ et selectionne 'Soumettre'")
    films = st.selectbox("Films : ", list_film_deroulante_films)
    submit_2 = st.form_submit_button("Soumettre")

if submit_2:
    df_film_choisi = df_fr[df_fr["originalTitle"] == films]

    if df_film_choisi.empty:
        st.error("No film found with the selected title.")
    else:
        film_choisi = df_film_choisi.iloc[:, 2:]

        # Debugging: Check the shape of the selected film's features
        #st.write("Shape of the selected film's features:", film_choisi.shape)
        
        film_choisi_scaled = scaler.transform(film_choisi)
        film_choisi_weighted = film_choisi_scaled * feature_weights

        # Find nearest neighbors
        distances, indices = nn.kneighbors(film_choisi_weighted)
        
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Debugging: Show distances and indices
        #st.write("Distances:", distances)
        #st.write("Indices:", indices)

        tconst = df_fr.iloc[indices.flatten(), 0].values
        suggestion = df_fr.iloc[indices.flatten(), 1].values

        col1 = st.columns(3)
        for film_suggestion, code_film, colonnes in zip(suggestion, tconst, col1):
            with colonnes:
                url = url_api + str(code_film) + key_api
                url_imdb2 = url_imdb + str(code_film)
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()
                    url_image = data.get('Poster', None)
                    if url_image:
                        st.image(url_image, width=200)
                    else:
                        st.write("No poster available.")
                except requests.exceptions.RequestException as e:
                    st.write(f"Error fetching data for {film_suggestion}: {e}")

                if isinstance(film_suggestion, str):
                    st.write(f" - [{film_suggestion}]({url_imdb2})")
                else:
                    st.write(f" - [{film_suggestion}]({url_imdb2})")

