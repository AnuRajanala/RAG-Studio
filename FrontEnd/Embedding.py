import streamlit as st
import pandas as pd
import numpy as np
import io

from BackEnd import EmbeddingStrategies
def embedding(next_page,prev_page):

    #st.header("Step 3: Embedding Generator")
    st.title("Embedding Generator")
    st.markdown("Embed document chunks and a user query using a selected Cohere model.")

    # ---------------------- UI Inputs ----------------------

    # Model selection
    embedding_models = [
        "embed-english-light-v2.0",
        "embed-english-light-v3.0",
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
        "embed-multilingual-light-v3.0"
    ]
    model = st.selectbox("🔎 Select Embedding Model", embedding_models)

    # User query input
    st.subheader("💬 User Query")
    query = st.text_input("Enter your query to embed", placeholder="e.g., What does Cohere do?")

    EmbeddingStrategies.trigger_embeddings(model,query)

    if st.button("⬅️ Back", key="back3"):
        prev_page()    with col3:
    if st.button("Next ➡️", key="next3"):
        next_page()
