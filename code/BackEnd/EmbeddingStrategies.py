import streamlit as st
import cohere
import requests
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def config():  
    COHERE_API_KEY = "i7WWTyDqa1YzVHo2SL2hKSyyU8rpYDYTb9pbmSfg"
    co = cohere.Client(COHERE_API_KEY)
    return co


def embedChunks(co):
    # Embed chunks
    """Embed document chunks and user query, store in session, and display."""
    input_type = "text" if st.session_state.model == "embed-english-light-v2.0" else "search_document"

    # Retrieve chunks from session (optional fallback)
    chunks = st.session_state.get("chunks", [])
    if chunks:
        chunk_response = co.embed(texts=chunks, model=st.session_state.model, input_type=input_type)
        chunk_embeddings = chunk_response.embeddings
        st.session_state["embeddings"] = chunk_embeddings 


def embedQuery(co):
    # Embed query
    input_type = "text" if st.session_state.model == "embed-english-light-v2.0" else "search_document"

    if st.session_state.query.strip():   
        query_input_type = "search_query" if input_type == "search_document" else input_type
        query_response = co.embed(texts=[st.session_state.query], model=st.session_state.model, input_type=query_input_type)
        query_embedding = query_response.embeddings
        st.session_state["query_embedding"] = query_embedding

def reduce_dimensions(embeddings, n_components=2):
    """Reduces the dimensionality of the embeddings using PCA."""
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    # """Reduces the dimensionality of the embeddings using t-SNE."""
    # tsne = TSNE(n_components=n_components, perplexity=1)
    # reduced_embeddings = tsne.fit_transform(embeddings)

    return reduced_embeddings


def generate_chunk_embeddings():
    if st.session_state.chunks:
        st.subheader("📌 Source Embeddings")
        for i, (chunk, embedding) in enumerate(zip(st.session_state.chunks, st.session_state["embeddings"])):
            with st.expander(f"Chunk {i+1}"):
                st.text_area("Text", chunk, height=80, disabled=True)
                st.json(embedding)
    else:
        st.info("ℹ️ No chunks provided. Only query embedding will be generated.")

    
def generate_query_embeddings():    
    if st.session_state.query.strip(): 
        st.subheader("🔍 Query Embedding")
        st.text_area("Query", st.session_state.query, height=80, disabled=True)
        st.json(st.session_state.query_embedding)
    else:
        st.warning("⚠️ Please enter a user query to generate its embedding.")


def trigger_embeddings():
    if st.button("🚀 Generate Embeddings"):
        with st.spinner("Generating embeddings..."):
            try:
                co = config()
                if co is None:
                    return
                embedChunks(co)
                generate_chunk_embeddings()
                embedQuery(co)
                generate_query_embeddings()
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ HTTP error from Cohere API: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")