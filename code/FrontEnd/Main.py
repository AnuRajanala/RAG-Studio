import streamlit as st

st.set_page_config(layout="wide",page_title="RAG Studio", page_icon=":red_circle:" ) 
import Chunking
import Ingest
import Embedding
import Visualize
import SimilaritySearches


pages = ["Ingestion", "Chunking", "Embedding", "SimilaritySearches"]

 
# Step/page initialization
if "page" not in st.session_state:
    st.session_state.page = 0

def next_page():
    if st.session_state.page < len(pages) - 1:
        st.session_state.page += 1
        st.rerun()  
def prev_page():
    if st.session_state.page > 0:
        st.session_state.page -= 1
        st.rerun()  

def barRendering(current, pages):
    bar = '<div style="width:100%; display:flex; justify-content:space-between; align-items:center; padding:8px; background-color:transparent;">'
    for id, descr in enumerate(pages):
        if id == current:
            bar += f'<span style="padding:8px; color:white; background-color:blue; border-radius:8px; flex-grow:1; text-align:center;">{descr}</span>'
        else:
            bar += f'<span style="padding:8px; color:white; background-color:#5ea5f7; flex-grow:1; text-align:center; border-radius:8px;">{descr}</span>'
        if id != len(pages) - 1:
            bar += '<span style="flex-grow:0; padding:0 8px; color:grey;">→</span>'
    bar += '</div>'
    st.markdown(bar, unsafe_allow_html=True)

def pageRendering(next_page, prev_page):
    barRendering(st.session_state.page, pages)
    if st.session_state.page == 0:
        Ingest.initialize()
        Ingest.ingest(next_page)

    elif st.session_state.page == 1:
        Chunking.initialize()
        Chunking.chunking(next_page, prev_page)

    elif st.session_state.page == 2:
        Embedding.initialize()
        Visualize.initialize()
        Embedding.embedding(next_page, prev_page)
        
    elif st.session_state.page == 3:
        SimilaritySearches.perform_similarity_search(prev_page)

st.title("RAG Studio")
pageRendering(next_page, prev_page)
