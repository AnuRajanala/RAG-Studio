import streamlit as st

def initialize():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    if "uploaded_file_content" not in st.session_state:
        st.session_state.uploaded_file_content = ""


def ingest(next_page):
    st.header("Step 1: Ingestion")

    file = st.file_uploader("Upload a file", key="file_uploader")
    if file is not None:
        st.session_state.uploaded_file = file
        st.session_state.uploaded_filename = file.name

        # Try reading file content if it's a text file
        try:
            content = file.read().decode('utf-8')
            st.session_state.uploaded_file_content = content
        except Exception:
            st.session_state.uploaded_file_content = "(Binary or unreadable file type)"

        st.success(f"Uploaded: {file.name}")
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