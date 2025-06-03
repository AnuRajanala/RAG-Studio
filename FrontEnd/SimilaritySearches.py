import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import CompareSearches

num_samples = 100
dimensionality = 5

st.set_page_config(layout="wide")
def perform_similarity_search(prev_page):
    with st.container():
        st.write('<h2 style="text-align: center;">Get Nearest Neighbors</h2>', unsafe_allow_html=True)
        search_options = ['Euclidean', 'Manhattan', 'Cosine','Cityblock','L1','L2','NaN_Euclidean']
        search = st.selectbox('Choose Similarity Metric Type', options = search_options, index = 0, key="limited_dropdown",help="Similarity Metric is used to measure distance between vectors." )
        # (Number of Nearest Neighbors to retrieve)
        topKValue = st.slider('Top K/limit', 1, 20,1,help="This parameter specifies how many similar vectors to retrieve for a given query.")
        #radius = st.text_input("Radius (for Range Search)")
        radius = st.number_input("Radius (for Range Search)",min_value=0.0, max_value=10.0,help="Range search (also known as radius search) retrieves all vectors that lie within a specified distance (or similarity range) from a query vector.")
        # Two action buttons
        cola, colb = st.columns(2)

        with cola:
            submit_search = st.button("Submit", type="primary", use_container_width=True)

        with colb:
            compare_searches = st.button("Compare All", type="secondary", use_container_width=True)

        
        if submit_search:
            with st.spinner("🔍 Generating Similarity Score..."):
                data, query_vector = generate_random_vectors(num_samples,dimensionality)
                perform_search(search,data, query_vector,topKValue)
        
        if compare_searches:
            with st.spinner("🔍 Comparing Searches..."):
                data, query_vector = generate_random_vectors(num_samples,dimensionality)
            with st.container():
                print_input_vector(data, query_vector)
            with st.container():
                CompareSearches.compare_searches(search_options,data, query_vector,topKValue) 
 
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        if st.button("⬅️ Back", key="back4"):
            prev_page()
    with col3:
        if st.button("Next ➡️", key="next4"):
            next_page()
            
    
def generate_random_vectors(num_vectors,dimensionality):
    data = np.random.rand(num_vectors, dimensionality)
    # Define a query vector
    query_vector = np.random.rand(dimensionality)
    return data, query_vector

def print_input_vector(data, query_vector):
    # Using columns to display multiple outputs side by side
    colA, colB = st.columns(2)

    with colA.container(key="data_container", border=False):
        with st.expander(label="Data Vector"):
            df = pd.DataFrame(data)
            st.markdown(f'<div class="center">{df.to_html(index=False, header = False)}</div>', unsafe_allow_html=True)

    with colB.container(height=300,key="query_container", border=False):
        st.markdown('<h6 style="text-align: center;">User Query Vector</h6>', unsafe_allow_html=True)
        df1 = pd.DataFrame(query_vector)
        st.markdown(f'<div class="center">{df1.to_html(index=False, header = False)}</div>', unsafe_allow_html=True)
        
def perform_search(search,data, query_vector,topKValue):
    nearest_neighbor_search(search,data, query_vector,topKValue)
    st.write("\n")
    print_input_vector(data, query_vector)

def nearest_neighbor_search(searchType,data, query_vector,topKValue):
    metric = searchType.lower()
    nbrs = NearestNeighbors(n_neighbors=topKValue, algorithm='auto', metric=metric)
    nbrs.fit(data)
    # Find the nearest neighbors for the query vector
    nn_distances, indices = nbrs.kneighbors([query_vector])
    similar_vectors = nn_distances.flatten(), data[indices.flatten()]
    cola, colb = st.columns([1,1])
    with cola:
        st.write(f'<h6 style="text-align: center;">Top {topKValue} Vectors from {searchType.title()} Search</h6>', unsafe_allow_html=True)
    with colb:
        lst = [item.tolist() for item in similar_vectors[1]]
        dfg = pd.DataFrame(lst)
        st.markdown(f'<div class="center">{dfg.to_html(index=True, header = False)}</div>', unsafe_allow_html=True)
        
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
