import streamlit as st
import oci
from oci.generative_ai_inference.generative_ai_inference_client import GenerativeAiInferenceClient 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from oci.generative_ai_inference.models import (
    GenerateTextDetails,
    OnDemandServingMode,
    CohereLlmInferenceRequest,   # if you're using Cohere family models
)
import oci.generative_ai_inference.models as models
import json

ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaboqoghc43bbqvei5y4pmnzia36wuh7fpcxn6fia7pfyelyg4rj7a"


# Read from default OCI config profile
CONFIG_PROFILE = "DEFAULT"
oci_config = oci.config.from_file(profile_name=CONFIG_PROFILE)
llmClient = oci.generative_ai_inference.GenerativeAiInferenceClient(config=oci_config, service_endpoint=ENDPOINT, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
embedClient = GenerativeAiInferenceClient(config=oci_config, service_endpoint=ENDPOINT, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))




MODEL_MAP = {
    "cohere.command-r-08-2024": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyanrlpnq5ybfu5hnzarg7jomak3q6kyhkzjsl4qj24fyoq",
    "cohere.command-r-plus-08-2024": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyaodm6rdyxmdzlddweh4amobzoo4fatlao2pwnekexmosq"
    # add others here
}

def chunk_color(chunk_number):
    palette = [ "blue", "green", "red", "blue", "blue", "orange", "violet", "pink", "red", "grey"    
    ]
    return palette[(chunk_number - 1) % len(palette)]



def embedChunks(chunks,client):
    # Embed chunks
    
    if chunks:
        
        # Define the batch size
        chunk_embeddings = []
        batch_size = 96
        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=st.session_state.model) 
        #embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MODEL_ID) 
        embed_text_detail.truncate = "NONE"
        embed_text_detail.compartment_id = "ocid1.compartment.oc1..aaaaaaaaboqoghc43bbqvei5y4pmnzia36wuh7fpcxn6fia7pfyelyg4rj7a"
    
        # Loop through the chunks in batches
        for i in range(0, len(chunks), batch_size):
            # Create a new EmbedTextDetails object for each batch
            embed_text_detail.inputs = chunks[i:i + batch_size]
            
            # Call the embed_text method
            response = client.embed_text(embed_text_detail)
    
            # Print result (optional, can be removed if not needed for every batch)
            #print("**************************Embed Texts Result**************************")
            #print(response.data)
    
            # Append the embeddings to the chunk_embeddings list
            chunk_embeddings.extend(response.data.embeddings)

        # Store the embeddings in the session state
        return chunk_embeddings


def match_answer_sentences_with_embeddings(answer, chunks, client, threshold=0.9):
    """
    For each sentence, find the chunk with highest cosine similarity
    (above threshold), to highlight it.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    chunk_embeddings = embedChunks(chunks, client)
    sentence_embeddings = embedChunks(sentences, client)
    
    highlights = []
    for i, sent_emb in enumerate(sentence_embeddings):
        similarities = cosine_similarity([sent_emb], chunk_embeddings)[0]
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        if max_sim >= threshold:
            highlights.append( (sentences[i], max_idx+1) )  # 1-based chunk
        else:
            highlights.append( (sentences[i], None) )
    return highlights

def render_highlighted_response(sentence_chunk_pairs):
    """
    sentence_chunk_pairs: list of (sentence, chunk_number or None)
    returns HTML string with colored highlighting
    """
    html_parts = []
    for sentence, chunk_num in sentence_chunk_pairs:
        if chunk_num:
            color = chunk_color(chunk_num)
            html_parts.append(
                f"<span style='background-color: {color}; padding:2px; border-radius:4px;'>"
                f"{sentence} <sup>Chunk {chunk_num}</sup></span>"
            )
        else:
            html_parts.append(sentence)
    return " ".join(html_parts)

def get_llm_data():
    llm_data = st.session_state.get("llm_data", {})
    return llm_data.get("top_k_results", []), llm_data.get("user_query", "")


def call_llm(gen_model_name, query, top_k_results):
    chunks_with_numbers = "\n".join(
        [f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(top_k_results)]
    )
    
    prompt = f"""
        You are a strict information extraction system.

        Your task is to answer the user query using only the content provided in the chunks below.

        Rules:
        - Only use the chunks to answer the question.
        - Do not use any outside knowledge.
        - Do not include greetings and asking futher assistance.
        - Do not say things like "Sure", "Hope this helps", or "Let me know if you need more".
        - Do not repeat or rephrase the question.
        - Do not introduce the answer with phrases like "Based on the chunks" or "Here is the answer".
        - If the answer is not explicitly found in the chunks, respond with exactly:
         "I cannot find an answer in the given chunks."

        Your response must contain only the answer, or the fallback message above.

        User Query:
        {query}

        Top-K Retrieved Chunks:
        {chunks_with_numbers}

        Provide a helpful response strictly from the chunks:
        """
    if gen_model_name in ["cohere.command-r-08-2024", "cohere.command-r-plus-08-2024"]:
        model_ocid = MODEL_MAP.get(gen_model_name)
        chat_details = models.ChatDetails(   # in some SDKs this is GenerateChatDetails
            compartment_id=COMPARTMENT_ID,
            serving_mode=models.OnDemandServingMode(model_id=model_ocid),
            chat_request=models.CohereChatRequest(   # in some SDKs: CohereChatInferenceRequest
                message=prompt,
                temperature=0.0,
                max_tokens=300
            )
        )

        # Call the chat operation (method name depends on SDK)
        response = llmClient.chat(chat_details) 
      

        return response.data.chat_response.text
    else:
        return "❌ Unknown model selected."

def generate_llm_response_page(prev_page):
    st.title("Step 5: Final Response")

    top_k_results, user_query = get_llm_data()

    if not top_k_results or not user_query:
        st.warning("⚠️ No data found. Please run the similarity search first.")
        return

    # Select model
    model_name = st.selectbox("Choose Generation Model", [
    "cohere.command-r-08-2024",        # Legacy large model
    "cohere.command-r-plus-08-2024"   # Legacy lightweight model
    ])

    # Show Top-K results
    top_k = st.session_state.get("topK")
    st.markdown(f"#### 🔍 Top-{top_k} Chunks")
    for i, res in enumerate(top_k_results, 1):
        st.markdown(f"**{i}.** {res}")

    # Submit Button: generates LLM response
    generate_clicked = st.button("🚀 Generate LLM Response")

    response = None
    if generate_clicked:
        with st.spinner("Generating response..."):
            response = call_llm(model_name, user_query, top_k_results)

    # Always render this at the bottom AFTER generation

    if generate_clicked and response:
        with st.spinner("Matching answer to chunks..."):
            highlights = match_answer_sentences_with_embeddings(response, top_k_results, embedClient, threshold=0.4)
        highlighted_html = render_highlighted_response(highlights)

        st.markdown("---")
        st.markdown("### ✨ LLM Response with Chunk Highlighting")
        st.markdown(highlighted_html, unsafe_allow_html=True)

    # Navigation buttons
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("⬅️ Back"):
            prev_page()