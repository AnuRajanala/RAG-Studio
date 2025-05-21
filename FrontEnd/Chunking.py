import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
print("CWD:", os.getcwd())
print("sys.path:", sys.path)
from st_aggrid import AgGrid, GridOptionsBuilder

from BackEnd import ChunkingStrategies


def initialize():
    if "chunking_strategy" not in st.session_state:
        st.session_state.chunking_strategy = ""

def longest_palindrome(s):
    def expand_around_center(s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    longest = ""
    for i in range(len(s)):
        # odd length palindrome
        odd_palindrome = expand_around_center(s, i, i)
        if len(odd_palindrome) > len(longest):
            longest = odd_palindrome

        # even length palindrome
        even_palindrome = expand_around_center(s, i, i + 1)
        if len(even_palindrome) > len(longest):
            longest = even_palindrome

    return longest

def reverseWords(s):
    
    words = s.split()
    reversed_words = [word[::-1] for word in words]
    return ' '.join(reversed_words)

def chunking(next_page, prev_page):
    st.header("Step 2: Chunking Strategy")
    #st.subheader("Select a Strategy")
    panel1, panel2, panel3 = st.columns(spec=[0.4,0.1,0.5],vertical_alignment="center", border=False)
    with panel1:
        with st.form('chunk'):
            chunking_strategy = st.selectbox('Select a Strategy:', ['Recursive', 'Semantic', 'Fixed'])
            chunkSize = st.number_input("Chunk Size", value=0)
            chunkOverlap = st.number_input("Chunk Overlap", value=0)
            submitted = st.form_submit_button('Submit',use_container_width=True)
    with panel2:
        if submitted:
            st.markdown("<h1 style='text-align: center;'>&rarr;</h1>", unsafe_allow_html=True)
    with panel3:
        if submitted:
            st.session_state.chunking_strategy = chunking_strategy
            #st.write(f'You chose {chunking_strategy}, chunk size={chunkSize}, chunk overlap={chunkOverlap}')
        
            st.info(st.session_state.uploaded_filename)
            content = st.session_state.uploaded_file_content
            if content and content != "(Binary or unreadable file type)":
                #st.code(content[:500], language='text')
                chunks = []
                if chunking_strategy == 'Recursive':
                    chunks = ChunkingStrategies.recursiveChunking(content, chunkSize, chunkOverlap)
                elif chunking_strategy == 'Fixed':
                    chunks = ChunkingStrategies.fixedChunking(content, chunkSize, chunkOverlap)

                if chunking_strategy != "":
                    colors = [
                        "red", "orange", "yellow", "green", "blue", "indigo", "violet",
                        "teal", "magenta", "brown", "gold", "lime", "navy", "coral"
                    ]
                    
                    # Display DataFrame as a table
                    words=[]
                    final_chunks=[]
                    embed_chunks=[]
                    for i, chunk in enumerate(chunks):
                        embed_chunks.append(chunk.page_content)
                        if words and words[1] != '':
                            chunk.page_content = f'<span style="color:red">{words[1]}</span>' + chunk.page_content[len(words[1]):]
                            words=[]
                            
                        #color = colors[i % len(colors)]
                        if i < len(chunks) - 1:
                            next_chunk = chunks[i + 1]
                            #chunk_words = set(chunk.page_content.split())
                            #next_chunk_words = set(next_chunk.page_content.split())
                            
                            overlap=chunk.page_content[-chunkOverlap:]
                            overlap_reversed = "".join(reverseWords(overlap))
                            overlapNext = next_chunk.page_content[:chunkOverlap] # get the last 10 characters of the current chunk
                            isPalindrome = overlap_reversed+ '      '+(overlapNext)
                            checkOverlap = longest_palindrome(isPalindrome)
                            words=checkOverlap.split('      ')
                            index = chunk.page_content.rfind(words[1].strip())
                            if words[1] != '' and index != -1:
                                # Color the substring
                                final_chunks.append(chunk.page_content[:index] + f'<span style="color:red">{words[1]}</span>' + chunk.page_content[index + len(words[1]):])
                            
                            else:
                                final_chunks.append(chunk.page_content)
                        else: 
                            final_chunks.append(chunk.page_content)
                    st.session_state["chunks"] = embed_chunks
                    df = pd.DataFrame({"Chunk": final_chunks})
                    st.session_state["df"] = df 
                    st.session_state["final_chunks"] = final_chunks 
                    
                     # Store the DataFrame in session state
                    #st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
                        #st.markdown(final_chunks, unsafe_allow_html=True)
                    
                    #df= pd.DataFrame({"Chunk" : final_chunks})
                    #gb = GridOptionsBuilder.from_dataframe(df)
                    #gb.configure_column("Chunk", header_name='Chunk', headerAlign="center", cellStyle={'textAlign': 'left'}, resizable=True)
                    #, autoHeight=True,                     wrapText=True, suppressHeaderMenuButton=True)

                    #gb.configure_column("No", headerName='No', width=50, pinned='left')
                    #gb.configure_grid_options(domLayout='autoHeight')
                    #gb.configure_pagination(enabled=True, paginationAutoPageSize=True, paginationPageSize=10)
                    #gb.configure_first_column_as_index(suppressMenu=True,sortable=False)
                    
                    #grid_options = gb.build()
                    
                    #AgGrid(df, gridOptions=grid_options, enable_enterprise_modules=True, fit_columns_on_grid_load=True)
            #else:
            #    st.write(content)
                    # Pagination
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
    #df.index += 1
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
    st.markdown(chunks[st.session_state.table_page].to_html(escape=False,justify='left'), unsafe_allow_html=True)