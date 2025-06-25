import streamlit as st
import pandas as pd
from BackEnd import ChunkingStrategies


def initialize():
    if "chunking_strategy" not in st.session_state:
        st.session_state.chunking_strategy = ""

def common_substring(statement1, statement2):
    for i in range(len(statement1), 0, -1):
        substring = statement1[-i:]
        if statement2.startswith(substring):
            return substring
    return ""


def chunking(next_page, prev_page):
    st.header("Step 2: Chunking Strategy")
    #st.subheader("Select a Strategy")
    panel1, panel2, panel3 = st.columns(spec=[0.4,0.1,0.5],vertical_alignment="center", border=False)
    with panel1:
        chunking_strategy = st.selectbox('Select a Strategy:', ['Recursive', 'Semantic', 'Fixed'])
        #with st.form('chunk'):
        embeddingStrategy = ''
        if chunking_strategy == 'Semantic':
            embedding_models = [
                    "embed-english-light-v2.0",
                    "embed-english-light-v3.0",
                    "embed-english-v3.0",
                    "embed-multilingual-v3.0",
                    "embed-multilingual-light-v3.0"
                ]
            embeddingStrategy = st.selectbox('Select an Embedding Strategy:', embedding_models)
            submitted = st.button('Submit',use_container_width=True)
        else:
            chunkSize = st.number_input("Chunk Size", value=0)
            chunkOverlap = st.number_input("Chunk Overlap", value=0)
            if chunkOverlap > chunkSize:
                st.error("Chunk overlap must be less than Chunk Size")
                submitted = st.button('Submit',use_container_width=True, disabled=True)
            else:
                submitted = st.button('Submit',use_container_width=True)
    with panel2:
        if submitted:
            st.markdown("<h1 style='text-align: center;'>&rarr;</h1>", unsafe_allow_html=True)
    with panel3:
        if submitted:
            st.session_state.chunking_strategy = chunking_strategy
            st.session_state.embeddingStrategy = embeddingStrategy
        
            st.info(st.session_state.uploaded_filename)
            content = st.session_state.uploaded_file_content
            if content and content != "(Binary or unreadable file type)":
                #st.code(content[:500], language='text')
                chunks = []
                if chunking_strategy == 'Recursive':
                    chunks = ChunkingStrategies.recursiveChunking(content, chunkSize, chunkOverlap)
                elif chunking_strategy == 'Fixed':
                    chunks = ChunkingStrategies.fixedChunking(content, chunkSize, chunkOverlap)
                elif chunking_strategy == 'Semantic':
                    chunks = ChunkingStrategies.semanticChunking(content, embeddingStrategy)

                if chunking_strategy != "":
                                        
                    final_chunks=[]
                    embed_chunks=[]
                    checkOverlap=''
                    colors = [
                        "red", "orange", "yellow", "green", "blue", "indigo", "violet",
                        "teal", "magenta", "brown", "gold", "lime", "navy", "coral"
                        ]
                    
                    for i, chunk in enumerate(chunks):
                        embed_chunks.append(chunk.page_content)
                        if chunking_strategy == 'Semantic':
                            continue
                        
                        if checkOverlap != '':
                            chunk.page_content = f'<span style="color:{color}">{checkOverlap}</span>' + chunk.page_content[len(checkOverlap):]

                        color = colors[i % len(colors)]

                        if i < len(chunks) - 1:
                            next_chunk = chunks[i + 1]                           
                            checkOverlap = common_substring(chunk.page_content, next_chunk.page_content)
                            index = chunk.page_content.rfind(checkOverlap)
                            if checkOverlap != '' and index != -1:
                                final_chunks.append(chunk.page_content[:index] + f'<span style="color:{color}">{checkOverlap}</span>' + chunk.page_content[index + len(checkOverlap):])
                            else:
                                final_chunks.append(chunk.page_content)
                        else: 
                            final_chunks.append(chunk.page_content)

                    st.session_state["chunks"] = embed_chunks
                    if chunking_strategy == 'Semantic':
                        final_chunks = embed_chunks
                    df = pd.DataFrame({"Chunk": final_chunks})
                    st.session_state["df"] = df 
                    st.session_state["final_chunks"] = final_chunks 
                    
        if 'df' in st.session_state:
            table_page()
            #cssGrid()              

    col1, col2, col3 = st.columns([1,9,1])
    with col1:
        if st.button("⬅️ Back", key="back2"):
            prev_page()
    with col3:
        if st.button("Next ➡️", key="next2"):
            next_page()

def cssGrid():
    final_chunks= st.session_state["final_chunks"]
    st.markdown("""
                <style>
                .scroll-box {
                    border: 1px solid #ccc;
                    padding: 8px;
                    height: 80px;
                    width: 100%;
                    overflow-y: scroll;
                    background-color: #F0F0F0;
                    color: #000000;
                    font-family: monospace;
                    white-space: pre-wrap;
                    border-radius: 6px;
                    margin-bottom: 10px;
                }
                </style>
            """, unsafe_allow_html=True)
    items_per_page = 10
    total_pages = (len(final_chunks) - 1) // items_per_page + 1
    if 'table_page' not in st.session_state:
        st.session_state.table_page = 0
            #with right:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(":arrow_left: Prev") and st.session_state.table_page > 0:
            st.session_state.table_page -= 1
    with col2:
        if st.button("Next :arrow_right:") and st.session_state.table_page < total_pages - 1:
            st.session_state.table_page += 1
    st.write(f"Page {st.session_state.table_page + 1} of {total_pages}")
    start = st.session_state.table_page * items_per_page
    end = start + items_per_page
    current_chunks = final_chunks[start:end]
    for c in current_chunks:
        st.markdown(f'<div class="scroll-box">{c}</div>', unsafe_allow_html=True)

def table_page():
    df = st.session_state["df"]
    page_size = 10
    chunks = [df[p:p + page_size] for p in range(0, len(df), page_size)]

    if 'table_page' not in st.session_state:
        st.session_state.table_page = 0

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("←"):
            st.session_state.table_page = max(0, st.session_state.table_page - 1)
    with col2:
        st.write(f"Page {st.session_state.table_page + 1} of {len(chunks)}")
    with col3:
        if st.button("→"):
            st.session_state.table_page = min(len(chunks) - 1, st.session_state.table_page + 1)
    st.write(f"**Total chunks:** {len(df)}")
    current_chunk = chunks[st.session_state.table_page].copy()
    start_index = st.session_state.table_page * page_size + 1
    current_chunk.index = range(start_index, start_index + len(current_chunk))
    st.markdown(current_chunk.to_html(escape=False, justify='left', classes='table'), unsafe_allow_html=True)
    st.markdown("""
        <style>
            .table td:not(:first-child) {
                word-wrap: break-word;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True)