import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

from BackEnd import ReduceEmbeddings

def initialize():
    if "selected_text" not in st.session_state:
        st.session_state.selected_text = ""
    

def visualize():
    st.subheader("📌Output vector projection")
    source_embeddings  = st.session_state["embeddings"]
    query_embedding = st.session_state["query_embedding"]
    combined_embeddings = np.concatenate((source_embeddings, query_embedding))
    combined_reduced = ReduceEmbeddings.reduce_dimensions(combined_embeddings)
    df = plotting(combined_reduced, len(source_embeddings))


def plotting(reduced_embeddings, query_index):
    print("Plotting started...")
    # Create a DataFrame for Plotly
    df = pd.DataFrame(reduced_embeddings, columns=['Dimension 1', 'Dimension 2'])
    df['Type'] = ['Source Embeddings'] * query_index + ['Query Embedding'] * (len(reduced_embeddings) - query_index)
    df['Index'] = range(1, len(reduced_embeddings) + 1)
    df['Embedding'] = [str(embedding) for embedding in np.concatenate((st.session_state["embeddings"], st.session_state["query_embedding"]))]
    fig = px.scatter(
        df, 
        x='Dimension 1', 
        y='Dimension 2', 
        color='Type', 
        color_discrete_map={'Source Embeddings': 'yellow', 
                            'Query Embedding': 'red'}, 
        hover_name='Index', 
        hover_data=['Embedding'], 
        text='Index')
    
    # Update the layout
    fig.update_layout(title='Visualizing embeddings in 2D space', xaxis_title='Dimension 1', yaxis_title='Dimension 2')
    fig.update_traces(textposition='top center')

    # Display the Plotly figure
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        clicked_point = st.plotly_chart(fig, use_container_width=True, key='plot')
    return df


def custom_plotting(embeddings,query_embedding,reduced_embeddings, query_index,key_name):
    print("Plotting started...")
    # Create a DataFrame for Plotly
    df = pd.DataFrame(reduced_embeddings, columns=['Dimension 1', 'Dimension 2'])
    df['Type'] = ['Source Embeddings'] * query_index + ['Query Embedding'] * (len(reduced_embeddings) - query_index)
    df['Index'] = range(1, len(reduced_embeddings) + 1)
    df['Embedding'] = [str(embedding) for embedding in np.concatenate((embeddings, query_embedding))]
    fig = px.scatter(
        df, 
        x='Dimension 1', 
        y='Dimension 2', 
        color='Type', 
        color_discrete_map={'Source Embeddings': 'yellow', 
                            'Query Embedding': 'red'}, 
        hover_name='Index', 
        hover_data=['Embedding'], 
        text='Index')
    
    # Update the layout
    fig.update_layout(title='Visualizing embeddings in 2D space', xaxis_title='Dimension 1', yaxis_title='Dimension 2')
    fig.update_traces(textposition='top center')

    # Display the Plotly figure
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        clicked_point = st.plotly_chart(fig, use_container_width=True, key=key_name)
    return df
