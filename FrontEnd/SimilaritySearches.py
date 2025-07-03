import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsRegressor
from scipy.spatial import distance
import CompareSearches
import Embedding
from BackEnd import EmbeddingStrategies
from BackEnd import ReduceEmbeddings
from FrontEnd import Visualize
from decimal import Decimal

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
    st.title("Step 5: Get TopK Nearest Neighbors")
    with st.container():
        #st.write('<h2 style="text-align: center;">Get Nearest Neighbors</h2>', unsafe_allow_html=True)
        search_options = ['Euclidean', 'Manhattan', 'Cosine','Cityblock','L1','L2','NaN_Euclidean']
        search = st.selectbox('Choose Similarity Metric Type', options = search_options, index = 0, key="limited_dropdown",help="Similarity Metric is used to measure distance between vectors." )
        # (Number of Nearest Neighbors to retrieve)
        topKValue = st.slider('Top K/limit', 1, 20,1,help="This parameter specifies how many similar vectors to retrieve for a given query.")
        #radius = st.text_input("Radius (for Range Search)")
        radius = st.number_input("Radius (for Range Search)",min_value=0.0, max_value=10.0,step = 0.10,help="Range search (also known as radius search) retrieves all vectors that lie within a specified distance (or similarity range) from a query vector.")
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
                perform_search(search,data, query_vector,topKValue,radius)
        
        if compare_searches:
            with st.spinner("🔍 Comparing Searches..."):
                query_vector = st.session_state["query_embedding"]
                data = st.session_state["embeddings"]
                CompareSearches.compare_searches(search_options,data, query_vector,topKValue,radius) 
 
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        if st.button("⬅️ Back", key="back4"):
            prev_page()
    with col3:
        if st.button("Next ➡️", key="next4"):
            next_page()
            
        
def perform_search(search,data, query_vector,topKValue,radius):
    nearest_neighbor_search(search,data, query_vector,topKValue,radius)


def nearest_neighbor_search(search_type, data, query_vector, top_k, radius):
    # Ensure data and query_vector are numpy arrays
    data = np.array(data)
    query_vector = np.array(query_vector).reshape(1, -1)  # Reshape query_vector to (1, 1024)
    st.write(f"after reshaping {query_vector}")

    # Flatten the data from (1, 3, 1024) to (3, 1024) for processing
    data = data.reshape(-1, 1024)

    # Determine the metric to use
    metric = search_type.lower().strip()

    nbrs = NearestNeighbors(n_neighbors=top_k, algorithm='auto', metric=metric)
    nbrs.fit(data)
    nn_distances, indices = nbrs.kneighbors(query_vector)
    
    nn_distances = nn_distances.flatten()
    indices = indices.flatten()

    # Filter by radius if specified
    if Decimal(f"{radius:.2f}")  > 0.00:
        if metric == 'cosine':
            valid_indices = np.where(1 - nn_distances >= radius)
            indices = indices[valid_indices]
        else:
            #indices = indices[nn_distances <= radius]
            valid_indices = np.where(nn_distances <= radius)
            indices = indices[valid_indices]
        # If no points are within the radius, return empty lists
        if len(indices) == 0:
            st.warning("No results found with similarity above the radius threshold.")
            return [], []
            
    # Extract the top-K vectors from the original data shape
    top_k_vectors = [data[i] for i in indices]

    # Display results using Streamlit
    cola, colb = st.columns([1, 1])
    with cola:
        st.write(f'<h6 style="text-align: center;">Top {top_k} Vectors from {search_type.title()} Search</h6>', unsafe_allow_html=True)
    with colb:
        vector_strings = []
        for i, vec in enumerate(top_k_vectors):
            st.write(f"Nearest Neighbor {i+1}: {vec}")
            vector_strings.append(str(vec))
            
    topk_embeddings = []
    topk_embeddings = top_k_vectors
    combined_embeddings = np.concatenate((topk_embeddings, query_vector))
    combined_reduced = ReduceEmbeddings.reduce_dimensions(combined_embeddings)
    Visualize.custom_plotting(topk_embeddings,query_vector,combined_reduced, len(topk_embeddings))

    original_chunks = st.session_state["chunks"]               # [N] (text)
    # Save to session_state for next page
    matched_text_chunks = [original_chunks[i] for i in indices.flatten()]
    if "llm_data" not in st.session_state:
        st.session_state.llm_data = {
            "top_k_results": [],  # Default value for top_k_results
            "user_query": ""      # Default value for user_query
        }
    st.session_state.llm_data["top_k_results"] = matched_text_chunks  # Example values
    st.session_state.llm_data["user_query"] = st.session_state.query  # Example query
    
    return top_k_vectors, nn_distances
        
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
