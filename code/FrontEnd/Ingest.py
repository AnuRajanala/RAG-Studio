import streamlit as st
from tika import parser
from markitdown import MarkItDown
import re

def initialize():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    if "uploaded_file_content" not in st.session_state:
        st.session_state.uploaded_file_content = ""


def ingest(next_page):
    st.header("Step 1: Ingestion")

    file = st.file_uploader("Upload a file", key="file_uploader",type=["pdf", "docx", "pptx", "txt"])

    if file is not None:
        st.session_state.uploaded_file = file
        st.session_state.uploaded_filename = file.name

        # Try reading file content if it's a text file
        try:
            #parsed_file = parser.from_file(file)
            md = MarkItDown(enable_plugins=False)
            result = md.convert(file)
            uploaded_file_content = result.markdown
            #uploaded_file_content = parsed_file['content']

            # Preprocess the text
            uploaded_file_content = uploaded_file_content.replace('\n', ' ')  # No change needed here, as '\n' is already a newline character
            uploaded_file_content = uploaded_file_content.replace('\\t', ' ')  # Replace tab characters with space
            #uploaded_file_content = re.sub(r'\s+', ' ', uploaded_file_content)  # Replace multiple whitespace characters with a single space
            #text = re.sub(r'([.!?])\s*', r'\1\n', text)  # Replace sentence-ending punctuation with a newline
            uploaded_file_content = re.sub(r'[^\w\s.!?-]', '', uploaded_file_content)  # Remove special characters except for punctuation and hyphen

            st.session_state.uploaded_file_content = uploaded_file_content
            #md = MarkItDown(enable_plugins=False)
            #result = md.convert(file)
            #st.session_state.uploaded_file_content = result.text_content

         
        except Exception:
            st.session_state.uploaded_file_content = "(Binary or unreadable file type)"

        st.success(f"Uploaded: {file.name}")
        st.success(f"Uploaded: {st.session_state.uploaded_file_content}")
    elif st.session_state.uploaded_filename:
        st.info(f"Previously uploaded: {st.session_state.uploaded_filename}")

    col1, col2, col3 = st.columns([1,9,1])
    with col3:
        #st.markdown("<div style='text-align: right;'></div>", unsafe_allow_html=True)
        if st.button("Next",icon="➡️", key="next1"):
            if st.session_state.uploaded_file is None and not st.session_state.uploaded_filename:
                st.error("You must upload a file to proceed.")
            else:
                next_page()