from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter as LangChainCharacterTextSplitter
from langchain.schema import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document as LlamaDocument
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import CohereEmbeddings
from langchain.schema import Document as LCDocument
import re
#import streamlit as st
COHERE_API_KEY = "eajAVMOvK5KtezTFd3AkOMSxbBePgkLuS0GFa2HF"
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

def preprocess_text(text):
    text = text.strip()
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text

def semanticChunking(uploaded_file_content, embeddingStrategy):
    llama_doc = LlamaDocument(text=preprocess_text(uploaded_file_content))
    cohere_lc = CohereEmbeddings(model=embeddingStrategy, cohere_api_key=COHERE_API_KEY, user_agent="cohere_embedding")
    embedding_model = LangchainEmbedding(cohere_lc)
    splitter = SemanticSplitterNodeParser(embed_model=embedding_model, buffer_size=1)
    nodes = splitter.get_nodes_from_documents([llama_doc])
    chunks = [LCDocument(page_content=node.text, metadata=node.metadata) for node in nodes]

    return chunks





