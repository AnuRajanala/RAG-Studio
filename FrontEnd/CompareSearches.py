import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

def compare_searches(search_options,data, query_vector,topKValue):
    catgory_new = []
    st.markdown('<h3 style="text-align: center;">{}</h3>'.format(f"Top {topKValue} Similar Vectors Comaprison:"), unsafe_allow_html=True)
    for search_type in search_options:
        catgory_new.append(search_type)
        fetch_comparison_results(search_type,data, query_vector,topKValue)
   
def fetch_comparison_results(searchType,data, query_vector,topKValue):
    nbrs = NearestNeighbors(n_neighbors=topKValue, algorithm='auto', metric=searchType.lower())
    nbrs.fit(data)
    # Find the nearest neighbors for the query vector
    nn_distances, indices = nbrs.kneighbors([query_vector])
    similar_vectors = nn_distances.flatten(), data[indices.flatten()]
    cola, coln, colb = st.columns([1,2,1])
    with cola:
        st.write(f'<h6 style="text-align: center;">{searchType.title()}</h6>', unsafe_allow_html=True)
    with coln:
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
