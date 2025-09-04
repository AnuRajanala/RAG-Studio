import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from BackEnd import ReduceEmbeddings

def initialize():
    if "radius" not in st.session_state:
          st.session_state.radius = 0.0


# Function to generate circle coordinates
def generate_circle(radius, center=(0, 0)):
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y


def points_within_circle(points, center, radius):
    distances = np.linalg.norm(points - center, axis=1)
    tolerance = 1e-6
    return distances <= radius + tolerance


def drawCircle(dataset, center_2d, fig, radius):
    # Generate circle coordinates based on selected radius
    x, y = generate_circle(st.session_state.radius, center=(center_2d[0], center_2d[1]))

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_shape='spline', fill='toself', name='Circle', line_color='lightcoral'))

    # Update the circle trace
    fig.data[2].x = x
    fig.data[2].y = y

    # Reorder the traces to have the circle on bottom
    fig.data = [fig.data[2], fig.data[0], fig.data[1]]      
    return fig


def visualize():
    st.subheader("📌Output vector projection")
    source_embeddings  = st.session_state["embeddings"]
    query_embedding = st.session_state["query_embedding"]

    source_reduced = ReduceEmbeddings.reduce_dimensions(source_embeddings)

    combined_embeddings = np.concatenate((source_embeddings, query_embedding))
    combined_reduced = ReduceEmbeddings.reduce_dimensions(combined_embeddings)

    plotting(source_reduced,combined_reduced, len(source_embeddings))
    return 'Plotted'

# The plotting function can be used to visualize the similarity between a query embedding and a set of source embeddings in a 2D space. 
# By adjusting the slider, users can explore the neighborhood around the query embedding and identify relevant chunks. 
# The checkbox allows users to toggle between displaying all chunks and displaying only chunks within the selected radius.

def plotting(source_reduced, combined_reduced, query_index):
    print("Plotting started...")
    # Create a DataFrame for Plotly
    query_reduced = combined_reduced[len(combined_reduced)-1]

    concat_reduced = np.concatenate([source_reduced, [query_reduced]],  dtype=object)
    df = pd.DataFrame(concat_reduced, columns=['Dimension 1', 'Dimension 2'])
    
    df['Type'] = ['Source Embeddings'] * query_index + ['Query Embedding'] * (len(concat_reduced) - query_index)
    df['Index'] = range(1, len(concat_reduced) + 1)
    df['Embedding'] = [str(embedding) for embedding in np.concatenate((st.session_state["embeddings"], st.session_state["query_embedding"]))]

    # Create a Plotly figure
    fig = px.scatter(
        df, 
        x='Dimension 1', 
        y='Dimension 2', 
        color='Type', 
        color_discrete_map={'Source Embeddings': 'blue', 
                            'Query Embedding': 'red'}, 
        hover_name='Index', 
        hover_data=['Embedding'], 
        text='Index'
        )
    
    fig.update_traces(textfont_size=8)   
    # Update the layout
    fig.update_layout(title='Visualizing embeddings in 2D space', xaxis_title='Dimension 1', yaxis_title='Dimension 2')
     
    center_2d = combined_reduced[len(combined_reduced)-1]

    dataset = source_reduced

    col1, col2 = st.columns([4,1])
    with col1:
        st.session_state.radius = st.slider("Select distance from the query embedding", min_value=0.0, max_value=100.0, value=st.session_state.radius, step=0.1) 
    
        if st.session_state.radius != 0.0:
            st.success(f"Plotting circle with radius: {st.session_state.radius}") 
        st.session_state.result = drawCircle(dataset, center_2d, fig, st.session_state.radius)
   
    within_circle = points_within_circle(dataset, center_2d, st.session_state.radius)

    indices = np.where(within_circle)[0]

    chunks_within_circle = [st.session_state.final_chunks[i] for i in (indices)]

    col1, col2, col3 = st.columns([5,0.1,4.5])
    with col1:
        plt = st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        checkbox_status = st.checkbox("Show Only Chunks within the circle")
        
        if checkbox_status:
            st.write(f"**Nearby Chunks:** {len(chunks_within_circle)}")
            chunks_df = pd.DataFrame({
            'Datapoint': indices + 1,
            "Chunk": chunks_within_circle})
        else:
            st.write(f"**Total Chunks:** {len(st.session_state.final_chunks)}")
            chunks_df = pd.DataFrame({
            'Datapoint': range(1, len(st.session_state.final_chunks) + 1),
            "Chunk": st.session_state.final_chunks})
        st.dataframe(chunks_df, hide_index=True)   

#This function plots data points for TopK results.
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
