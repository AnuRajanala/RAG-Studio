import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from FrontEnd import Visualize
from decimal import Decimal
import Embedding
from BackEnd import EmbeddingStrategies
from BackEnd import ReduceEmbeddings
import CohereReranker

def compare_searches(search_options,data, query_vector,topKValue,radius):
    catgory_new = []
    types_list = []
    st.markdown('<h3 style="text-align: center;">{}</h3>'.format(f"Top {topKValue} Nearest Neighbors Comaprison:"), unsafe_allow_html=True)
    for search_type in search_options:
        catgory_new.append(search_type)
        types_list = compare_nearest_neighbor_searches(search_type,data, query_vector,topKValue,radius)
        combined_embeddings = np.concatenate((types_list, query_vector))
        combined_reduced = ReduceEmbeddings.reduce_dimensions(combined_embeddings)
        Visualize.custom_plotting(types_list,query_vector,combined_reduced, len(types_list),search_type)

   
def fetch_comparison_results(searchType,data, query_vector,topKValue,radius):
    nbrs = NearestNeighbors(n_neighbors=topKValue, algorithm='auto', metric=searchType.lower())
    nbrs.fit(data)
    # Find the nearest neighbors for the query vector
    nn_distances, indices = nbrs.kneighbors(query_vector)
    similar_vectors = nn_distances.flatten(), data[indices.flatten()]
    cola, coln, colb = st.columns([1,2,1])
    with cola:
        st.write(f'<h6 style="text-align: center;">{searchType.title()}</h6>', unsafe_allow_html=True)
    with coln:
        lst = [item.tolist() for item in similar_vectors[1]]
        st.write(similar_vectors[1])
        dfg = pd.DataFrame(lst)


def compare_nearest_neighbor_searches(search_type,data, query_vector,topKValue,radius):
    data = np.array(data)
    query_vector = np.array(query_vector)  # Reshape query_vector to (1, 1024)

    # Determine the metric to use
    metric = search_type.lower().strip()
    nbrs = NearestNeighbors(n_neighbors=topKValue, algorithm='auto', metric=metric)
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
        #st.write(f"Found {len(within_radius_indices)} points within radius")
        #st.write(f"Data points within radius:\n{within_radius_data_points}") 
        #st.write(f"Distances of these points: {within_radius_distances}")   
        topk_embeddings = []
        topk_embeddings = within_radius_data_points
        updated_indices = within_radius_indices
        # If no points are within the radius, return empty lists
        if len(within_radius_indices) == 0:
            st.warning("No results found with similarity above the radius threshold.")
            return [], [] 

    cola, coln, colb = st.columns([1,2,1])
    with cola:
        st.write(f'<h6 style="text-align: center;">{metric.title()}</h6>', unsafe_allow_html=True)
    with coln:
        #st.write(topk_embeddings)
        for i, vec in enumerate(topk_embeddings):
            st.write(f"Nearest Neighbor {i+1}: {vec}") 
            
    original_chunks = st.session_state["chunks"]               # [N] (text)
    # Save to session_state for next page
    matched_text_chunks = [original_chunks[i] for i in updated_indices.flatten()]
    if "llm_data" not in st.session_state:
        st.session_state.llm_data = {
            "top_k_results": [],  # Default value for top_k_results
            "user_query": ""      # Default value for user_query
        }
    st.session_state.llm_data["top_k_results"] = matched_text_chunks  # Example values
    st.session_state.llm_data["user_query"] = st.session_state.query  # Example query
    #CohereReranker.initialize()
    #CohereReranker.rerank_documents()
    
    return topk_embeddings

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
