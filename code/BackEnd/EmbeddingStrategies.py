import streamlit as st
import cohere
import requests
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import oci
from oci.generative_ai_inference.generative_ai_inference_client import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails
from oci.generative_ai_inference.models import OnDemandServingMode
import requests

ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaltks2v7drnqi6rtik7frrl4kypg2jz7ykkirwzbxgbbtzjsfnrrq"

def config():
    # Read from default OCI config profile
    CONFIG_PROFILE = "DEFAULT"
    oci_config = oci.config.from_file('C:/Users/ychavva/AITrial/.oci/config', CONFIG_PROFILE)
    client = GenerativeAiInferenceClient(config=oci_config, service_endpoint=ENDPOINT, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
    return client


def embedChunks(client):
    # Embed chunks
    chunks = st.session_state.get("chunks", [])
    if chunks:
        
        # Define the batch size
        chunk_embeddings = []
        batch_size = 96
        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=st.session_state.model) 
        #embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MODEL_ID) 
        embed_text_detail.truncate = "NONE"
        embed_text_detail.compartment_id = "ocid1.compartment.oc1..aaaaaaaaboqoghc43bbqvei5y4pmnzia36wuh7fpcxn6fia7pfyelyg4rj7a"
    
        # Loop through the chunks in batches
        for i in range(0, len(chunks), batch_size):
            # Create a new EmbedTextDetails object for each batch
            embed_text_detail.inputs = chunks[i:i + batch_size]
            
            # Call the embed_text method
            response = client.embed_text(embed_text_detail)
    
            # Print result (optional, can be removed if not needed for every batch)
            #print("**************************Embed Texts Result**************************")
            #print(response.data)
    
            # Append the embeddings to the chunk_embeddings list
            chunk_embeddings.extend(response.data.embeddings)

        # Store the embeddings in the session state
        st.session_state["embeddings"] = chunk_embeddings


def embedQuery(client):
    # Embed query
    if st.session_state.query.strip():
        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=st.session_state.model)
        embed_text_detail.inputs = [st.session_state.query]
        embed_text_detail.truncate = "NONE"
        embed_text_detail.compartment_id = "ocid1.compartment.oc1..aaaaaaaaboqoghc43bbqvei5y4pmnzia36wuh7fpcxn6fia7pfyelyg4rj7a"
        response = client.embed_text(embed_text_detail) 
        query_embedding = [e for e in response.data.embeddings]
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
                st.text_area("Text", chunk, height=80, disabled=True, key=f"text_chunk_{i}")
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
                st.error(f"❌ HTTP error from API: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
