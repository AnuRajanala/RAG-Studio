import streamlit as st
import cohere
import requests

def initialize():
    if "query_embedding" not in st.session_state:
        st.session_state["query_embedding"] = ""
    if "llm_data" not in st.session_state:
        st.session_state.llm_data = {
            "top_k_results": [],  # Default value for top_k_results
            "user_query": ""      # Default value for user_query
        }
        
def config():  
    COHERE_API_KEY = "Vi2dcrXuERFSLadpdwSC2lHclg7WFzXmyvA16LTW"
    co = cohere.Client(COHERE_API_KEY)
    return co

def rerank_documents(rerank_candidates):
    with st.spinner("Reranking Top K results..."):
        try:
            co = config()
            if co is None:
                return
            topKDocuments = st.session_state.llm_data["top_k_results"]
            user_query = st.session_state.llm_data["user_query"]
            response = co.rerank(model="rerank-v3.5",query=user_query,documents=topKDocuments,top_n=rerank_candidates)
            matched_text_chunks = []
            # Print top reranked results
            st.write(f'<h5 style="text-align: center;">Top {rerank_candidates} reranked documents:</h5>', unsafe_allow_html=True)
            for i, result in enumerate(response.results):
                st.write(f"<h6>Document {i + 1}.</h6> {topKDocuments[result.index]} <h5>(Score: {result.relevance_score:.4f})</h4>\n", unsafe_allow_html=True)
                matched_text_chunks.append(topKDocuments[result.index])
                
            if "llm_data" not in st.session_state:
                st.session_state.llm_data = {
                    "top_k_results": [],  # Default value for top_k_results
                    "user_query": ""      # Default value for user_query
                }
            st.session_state.llm_data["top_k_results"] = matched_text_chunks
            st.session_state.llm_data["user_query"] = st.session_state.query
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP error from Cohere API: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
