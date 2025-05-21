import streamlit as st
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Chunking
import Ingest
import Embedding

pages = ["Ingestion", "Chunking", "Embedding"]

st.set_page_config(layout="wide",page_title="RAG Studio", page_icon=":red_circle:" )  
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
    bar = ""
    for id, descr in enumerate(pages):
        if id == current:
            bar += f'<span style="padding:8px; color:white; background-color:#0073e6; border-radius:8px;">{descr}</span>'
        else:
            bar += f'<span style="padding:8px; color:#0073e6;">{descr}</span>'
        if id != len(pages) - 1:
            bar += " &rarr; "
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
        Embedding.embedding(prev_page)

st.title("RAG Studio")
pageRendering(next_page, prev_page)