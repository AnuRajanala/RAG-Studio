from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter as LangChainCharacterTextSplitter
from langchain.schema import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document as LlamaDocument
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import CohereEmbeddings
from langchain.schema import Document as LCDocument
import re
import oci
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
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

def preprocess_text(text):
    text = text.strip()
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text

def semanticChunking(uploaded_file_content, embeddingStrategy):
    file_content = Document(page_content=uploaded_file_content)
    embedding = OCIGenAIEmbeddings(
        model_id=embeddingStrategy,
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaboqoghc43bbqvei5y4pmnzia36wuh7fpcxn6fia7pfyelyg4rj7a"
    )
    text_splitter = LangChainSemanticChunker(embeddings = embedding #,
                                        # add_start_index = "true",
                                        # breakpoint_threshold_type  = "percentile",
                                        # breakpoint_threshold_amount = 100,
                                        # sentence_split_regex="(?<=[.?!])\\s+",
                                         #number_of_chunks = 100,
                                          #min_chunk_size= 100
                                         )
    chunks = text_splitter.split_documents([file_content])
    # llama_doc = LlamaDocument(text=preprocess_text(uploaded_file_content))
    # cohere_lc = CohereEmbeddings(model=embeddingStrategy, cohere_api_key=COHERE_API_KEY, user_agent="cohere_embedding")
    # embedding_model = LangchainEmbedding(cohere_lc)
    # splitter = SemanticSplitterNodeParser(embed_model=embedding_model, buffer_size=1)
    # nodes = splitter.get_nodes_from_documents([llama_doc])
    # chunks = [LCDocument(page_content=node.text, metadata=node.metadata) for node in nodes]

    return chunks





