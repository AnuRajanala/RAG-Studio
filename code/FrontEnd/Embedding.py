import streamlit as st
import pandas as pd
import numpy as np
import io
import time

from BackEnd import EmbeddingStrategies
from FrontEnd import Visualize

def initialize():
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "model" not in st.session_state:
        st.session_state.model = ""
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = ""
    if "query_embedding" not in st.session_state:
        st.session_state["query_embedding"] = ""

def callVisualize():
    st.session_state.is_plotted = Visualize.visualize()  

def embedding(next_page,prev_page):

    st.title("Step 3: Embedding Generator")
    st.markdown("Embed document chunks and a user query using a selected Cohere model.")

    # ---------------------- UI Inputs ----------------------

    # Model selection
    if st.session_state.chunking_strategy == 'Semantic':
        embedding_models = [st.session_state.embeddingStrategy]
    else:
        embedding_models = [
            "cohere.embed-english-v3.0",
            "cohere.embed-english-light-v3.0",
            "cohere.embed-multilingual-v3.0",
            "cohere.embed-multilingual-light-v3.0",
            "cohere.embed-english-light-v2.0"
        ]
    #model = st.selectbox("🔎 Select Embedding Model", embedding_models)

    st.subheader("🔎 Embedding Model")
    st.session_state.model = st.selectbox("Select embedding model to be used to generate embeddings", embedding_models)

    # User query input
    st.subheader("💬 User Query")
    st.session_state.query = st.text_input("Enter your query to embed", placeholder="e.g., What does Cohere do?")



    if st.session_state.query.strip(): 

        tab1, tab2 = st.tabs(["Projection", "Vectors"])
        css = '''
            <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.5rem;
            }
            </style>
            '''
        st.markdown(css, unsafe_allow_html=True)
        with tab1:
            co = EmbeddingStrategies.config()   
            EmbeddingStrategies.embedChunks(co)               
            EmbeddingStrategies.embedQuery(co)
            callVisualize()
        with tab2:
            EmbeddingStrategies.generate_chunk_embeddings()
            EmbeddingStrategies.generate_query_embeddings()
    else:
        st.warning("⚠️ Please enter a user query to generate its embedding.")



        col1, col2, col3 = st.columns([1,9,1])
        with col1:
            if st.button("⬅️ Back", key="back3"):
                prev_page()
        with col3:
            if st.button("Next ➡️", key="next3"):
                next_page()
