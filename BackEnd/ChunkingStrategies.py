from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter as LangChainCharacterTextSplitter
from langchain.schema import Document
#import streamlit as st

def recursiveChunking(uploaded_file_content, chunkSize, chunkOverlap):
    file_content = Document(page_content=uploaded_file_content)

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunkSize,   
        chunk_overlap=chunkOverlap
    )

    chunks = recursive_text_splitter.split_documents([file_content])
    return chunks

def fixedChunking(uploaded_file_content, chunkSize, chunkOverlap):
    file_content = Document(page_content=uploaded_file_content)

    fixed_text_splitter = LangChainCharacterTextSplitter(
        separator="",
        chunk_size=chunkSize,   
        chunk_overlap=chunkOverlap
    )

    chunks = fixed_text_splitter.split_documents([file_content])
    return chunks






