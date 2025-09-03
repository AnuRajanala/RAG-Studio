import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import CompareSearches
import Embedding
from BackEnd import EmbeddingStrategies
from BackEnd import ReduceEmbeddings
from FrontEnd import Visualize
from decimal import Decimal
import CohereReranker

st.set_page_config(layout="wide")
def initialize():
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = ""
    if "query_embedding" not in st.session_state:
        st.session_state["query_embedding"] = ""
    if "llm_data" not in st.session_state:
        st.session_state.llm_data = {
            "top_k_results": [],  # Default value for top_k_results
            "user_query": ""      # Default value for user_query
        }
        
def perform_similarity_search(next_page,prev_page):
    st.header("Step 5: Retrieval and Reranking of Top K Nearest Neighbors", divider=True)
    with st.container():
        search_options = ['Euclidean', 'Manhattan', 'Cosine','NaN_Euclidean']
        search = st.selectbox('Choose Similarity Metric Type', options = search_options, index = 0, key="limited_dropdown",help="Similarity Metric is used to measure distance between vectors." )
        # (Number of Nearest Neighbors to retrieve)
        topKValue = st.slider('Top K/limit', 1, 20,1,help="This parameter specifies how many similar vectors to retrieve for a given query.")
        radius = st.number_input("Radius (for Range Search)",min_value=0.000, max_value=100.000,format="%.5f",help="Range search (also known as radius search) retrieves all vectors that lie within a specified distance (or similarity range) from a query vector.")
        enable_rerank = st.checkbox("Enable Reranking", value=False,help="Reranking is used to reorder or refine a set of initially retrieved results based on their relevance to a user's query, gives the most relevant responses to a particular query.")
        rerank_candidates = st.slider("Number of candidates for reranking", min_value=2, max_value=20, value=5,help="More candidates = better reranking but slower")
        # Two action buttons
        cola, colb = st.columns(2)

        with cola:
            submit_search = st.button("Submit", type="primary", use_container_width=True)

        with colb:
            compare_searches = st.button("Compare All", type="secondary", use_container_width=True)
            
        if submit_search:
            with st.spinner("🔍 Generating Similarity Score..."):
                query_vector = st.session_state["query_embedding"]
                data = st.session_state["embeddings"]
                perform_search_and_rerank(search,data, query_vector,topKValue,radius,enable_rerank,rerank_candidates)
        if compare_searches:
            with st.spinner("🔍 Comparing Searches..."):
                query_vector = st.session_state["query_embedding"]
                data = st.session_state["embeddings"]
                CompareSearches.compare_searches(search_options,np.array(data), np.array(query_vector),topKValue,radius)
 
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        if st.button("⬅️ Back", key="back4"):
            prev_page()
    with col3:
        if st.button("Next ➡️", key="next4"):
            next_page()
            
def nearest_neighbor_search(search_type, data, query_vector, top_k, radius):
    data = np.array(data)
    query_vector = np.array(query_vector)  # Reshape query_vector to (1, 1024)

    # Determine the metric to use
    metric = search_type.lower().strip()
    nbrs = NearestNeighbors(n_neighbors=top_k, algorithm='auto', metric=metric)
    nbrs.fit(data)
    nn_distances, indices = nbrs.kneighbors(query_vector)
    similar_vectors = nn_distances.flatten(), data[indices.flatten()]
    
    topk_embeddings = []
    topk_embeddings = similar_vectors[1]
    updated_indices = indices
    # Filter by radius if specified
    if Decimal(f"{radius:.2f}")  > 0.00:
        within_radius_mask = nn_distances[0] <= radius
        within_radius_distances = nn_distances[0][within_radius_mask]
        within_radius_indices = indices[0][within_radius_mask]
        within_radius_data_points = data[within_radius_indices]
        st.write(f"Found {len(within_radius_indices)} points within radius")
        st.write(f"Data points within radius:\n{within_radius_data_points}") 
        st.write(f"Distances of these points: {within_radius_distances}")   
        topk_embeddings = []
        topk_embeddings = within_radius_data_points
        updated_indices = within_radius_indices
        # If no points are within the radius, return empty lists
        if len(within_radius_indices) == 0:
            st.warning("No results found with similarity above the radius threshold.")
            return [], [] 
            
    # Display results
    cola, colb = st.columns([1, 1])
    with cola:
        st.write(f'<h6 style="text-align: center;">Top {top_k} Nearest Neighbors from {search_type.title()} Search</h6>', unsafe_allow_html=True)
    with colb:
        for i, vec in enumerate(topk_embeddings):
            st.write(f"Nearest Neighbor {i+1}: {vec}")         
            
    st.write(f'<h6 style="text-align: center;">Top K Vectors plot without reranking</h6>', unsafe_allow_html=True)
    combined_embeddings = np.concatenate((topk_embeddings, query_vector))
    combined_reduced = ReduceEmbeddings.reduce_dimensions(combined_embeddings)
    Visualize.custom_plotting(topk_embeddings,query_vector,combined_reduced, len(topk_embeddings),metric)

    original_chunks = st.session_state["chunks"]
    # Save to session_state for next page
    matched_text_chunks = [original_chunks[i] for i in updated_indices.flatten()]
    if "llm_data" not in st.session_state:
        st.session_state.llm_data = {
            "top_k_results": [],  # Default value for top_k_results
            "user_query": ""      # Default value for user_query
        }

    st.session_state.llm_data["top_k_results"] = matched_text_chunks
    st.session_state.llm_data["user_query"] = st.session_state.query
    return topk_embeddings
    
def perform_search_and_rerank(search,data, query_vector,topKValue,radius,enable_rerank,rerank_candidates):
    topk_results = nearest_neighbor_search(search,data, query_vector,topKValue,radius)
    if enable_rerank and len(topk_results) > 1:
        st.write(f'<h5 style="text-align: center;">Top {topKValue} documents without reranking:</h5>', unsafe_allow_html=True)
        topKDocuments = st.session_state.llm_data["top_k_results"]
        for i, result in enumerate(topKDocuments):
                st.write(f"<h6> Document {i + 1}.</h6> {result}\n", unsafe_allow_html=True)
        CohereReranker.initialize()
        CohereReranker.rerank_documents(rerank_candidates)


st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        color: #AA7825;
    }
    </style>
""", unsafe_allow_html=True)
