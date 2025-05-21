import streamlit as st
import cohere
import requests

def generate_embeddings(model,query):
    # ---------------------- Configuration ----------------------
    COHERE_API_KEY = "eajAVMOvK5KtezTFd3AkOMSxbBePgkLuS0GFa2HF"
    co = cohere.Client(COHERE_API_KEY)

    #model: str, chunks: list[str], query: str
    
    #st.set_page_config(page_title="Embedding Generator", layout="centered")
    

    # Retrieve chunks from session (optional fallback)
    chunks = st.session_state.get("chunks", [])

    # ---------------------- Main Method ----------------------


    """Embed document chunks and user query, store in session, and display."""
    input_type = "text" if model == "embed-english-light-v2.0" else "search_document"

    # Embed chunks
    if chunks:
        chunk_response = co.embed(texts=chunks, model=model, input_type=input_type)
        chunk_embeddings = chunk_response.embeddings
        st.session_state["embeddings"] = chunk_embeddings

        st.subheader("📌 Chunk Embeddings")
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            with st.expander(f"Chunk {i+1}"):
                st.text_area("Text", chunk, height=80, disabled=True)
                st.json(embedding)
    else:
        st.info("ℹ️ No chunks provided. Only query embedding will be generated.")

    # Embed query
    if query.strip():
        query_input_type = "search_query" if input_type == "search_document" else input_type
        query_response = co.embed(texts=[query], model=model, input_type=query_input_type)
        query_embedding = query_response.embeddings[0]
        st.session_state["query_embedding"] = query_embedding

        st.subheader("🔍 Query Embedding")
        st.text_area("Query", query, height=80, disabled=True)
        st.json(query_embedding)
    else:
        st.warning("⚠️ Please enter a user query to generate its embedding.")

# ---------------------- Trigger ----------------------
def trigger_embeddings(model,query):
    if st.button("🚀 Generate Embeddings"):
        with st.spinner("Generating embeddings..."):
            try:
                generate_embeddings(model,query)
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ HTTP error from Cohere API: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")