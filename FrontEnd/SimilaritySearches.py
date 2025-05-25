import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from scipy.spatial.distance import cosine

num_samples = 100
dimensionality = 5


def perform_similarity_search(prev_page):
    with st.container():
        #"""Initialize Search Operations"""
        st.write('<h2 style="text-align: center;">Perform Similarity Searches</h2>', unsafe_allow_html=True)
        search_options = {
    "Euclidean Search": [],
    "Cosine Search": [],
    "Nearest Neighbors Search": ['euclidean', 'manhattan', 'cosine','cityblock','haversine','l1','l2','nan_euclidean']
}
        search = st.selectbox('Choose Similarity Metric Type', options = list(search_options.keys()), index = 0, key="limited_dropdown",help="Similarity Metric is used to measure distance between vectors." )
        if(search == 'Nearest Neighbors Search'):
            nnSearch = st.selectbox('Distance Metric Type for Nearest Neighbor Search:', options = search_options[search], index = 0 )
        # Number of Nearest Neighbors to retrieve
        topKValue = st.slider('Top K/limit', 1, 50,1,help="This parameter specifies how many similar vectors to retrieve for a given query.")
        #Radius (for Range Search)
        radius = st.number_input("Radius (for Range Search)",min_value=0.0, max_value=10.0,help="Range search (also known as radius search) retrieves all vectors that lie within a specified distance (or similarity range) from a query vector.")
        if(st.button('Submit',type="primary")):
            data, query_vector = generate_random_vectors(num_samples,dimensionality)
            perform_search(search,data, query_vector,topKValue)

    col1, col2, col3 = st.columns([1,9,1])
    with col1:
        if st.button("⬅️ Back", key="back2"):
            prev_page()
    with col3:
        if st.button("Next ➡️", key="next2"):
            next_page()
        
def generate_random_vectors(num_vectors,num_dimensions):
    data = np.random.rand(num_vectors, num_dimensions)
    from sklearn.metrics.pairwise import euclidean_distances
        
    # Define a query vector
    query_vector = np.random.rand(dimensionality)
    return data, query_vector

def print_input_vector(data, query_vector):
    # Using columns to display multiple outputs side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h6 style="text-align: center;">Data Vector</h6>', unsafe_allow_html=True)
        df = pd.DataFrame(data)
        st.markdown(f'<div class="center">{df.to_html(index=False)}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h6 style="text-align: center;">Input Query Vector</h6>', unsafe_allow_html=True)
        df = pd.DataFrame(query_vector)
        st.markdown(f'<div class="center">{df.to_html(index=False)}</div>', unsafe_allow_html=True)
        
def perform_search(search,data, query_vector,topKValue):
    if(search == 'Euclidean Search'):
        euclidean_search(data, query_vector)
        euclidean_search_topk(data, query_vector,topKValue)
        st.write("\n")
    if(search == 'Cosine Search'):
        cosine_search(data, query_vector)
        cosine_search_topk(data, query_vector)
    if(search == 'Nearest Neighbors Search'):
        nearest_neighbor_search(nnSearch,data, query_vector)
    print_input_vector(data, query_vector)

def euclidean_search(data, query_vector):
    distances = euclidean_distances(data, [query_vector])

    # Find the closest vector
    closest_index = np.argmin(distances)
    closest_vector = data[closest_index]
    st.write(f"Closest vector from Euclidean Search: {closest_vector}")
    
def euclidean_search_topk(data, query_vector,topKValue):
      # Calculate Euclidean distances between the query vector and all vectors
    distances = np.array([distance.euclidean(query_vector, vec) for vec in data])
    # Get the indices of the top-k smallest distances
    top_k_indices = np.argsort(distances)[:topKValue]

    # Get the corresponding distances
    top_k_distances = distances[top_k_indices]
    for idx in top_k_indices:
        formatted_vector = '[' + ' '.join(f'{x:.8f}' for x in data[idx]) + ']'
        st.markdown(f'<div class="center">{formatted_vector}</div>', unsafe_allow_html=True)
    
def cosine_search_topk(data, query_vector,topKValue):
    # Get the indices of the top-k similarities
    cosine_distances = np.array([distance.cosine(query_vector, vec) for vec in data])
    top_k_cosine = np.argsort(cosine_distances)[:topKValue]

    for idx in top_k_cosine:
        formatted_cosine_vector = '[' + ' '.join(f'{x:.8f}' for x in data[idx]) + ']'
        st.markdown(f'<div class="center">{formatted_cosine_vector}</div>', unsafe_allow_html=True)
        
def cosine_search(data, query_vector):
    similarities = cosine_similarity(data, [query_vector])
    most_similar_index = np.argmax(similarities)
    most_similar_vector = data[most_similar_index]
    st.write(f"Most similar vector from Cosine Similarity: {most_similar_vector}")
        
def find_similar_vectors(query_vector, data, n_neighbors=1):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
    nbrs.fit(data)
    # Find the nearest neighbors for the query vector
    nn_distances, indices = nbrs.kneighbors([query_vector])
    return nn_distances.flatten(), data[indices.flatten()]
        
def nearest_neighbor_search(searchType,data, query_vector,topKValue):
    nbrs = NearestNeighbors(n_neighbors=topKValue, algorithm='auto', metric=searchType)
    nbrs.fit(data)
    # Find the nearest neighbors for the query vector
    nn_distances, indices = nbrs.kneighbors([query_vector])
    similar_vectors = nn_distances.flatten(), data[indices.flatten()]
    st.markdown('<h6 style="text-align: center;">{}</h6>'.format(f"Top {topKValue} vectors from Nearest Neighbors Search:"), unsafe_allow_html=True)
    for idx in indices:
        f1 = '[' + '{}'.format(data[idx])+ ']'
        st.markdown(f'<div class="center">{f1}</div>', unsafe_allow_html=True)