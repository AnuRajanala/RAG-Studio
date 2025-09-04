import streamlit as st
import cohere
import re
from huggingface_hub import InferenceClient
import os
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np

COHERE_API_KEY = "i7WWTyDqa1YzVHo2SL2hKSyyU8rpYDYTb9pbmSfg"
HF_API_TOKEN = "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"

provider = "novita"

client = InferenceClient(
    provider=provider,
    api_key="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",
)
# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

def chunk_color(chunk_number):
    palette = [
        "#FFB6C1", "#ADD8E6", "#90EE90", "#FFD700", "#FFA07A", 
        "#DDA0DD", "#87CEFA", "#F0E68C", "#E6E6FA", "#FA8072",
        "#98FB98", "#B0C4DE", "#FFE4B5", "#AFEEEE", "#FFDEAD"
    ]
    return palette[(chunk_number - 1) % len(palette)]


def embed_sentences(sentences, co):
    response = co.embed(
        texts=sentences,
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return response.embeddings


def embed_chunks(chunks, co):
    """
    Returns list of embedding vectors for the chunks
    """
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings


def match_answer_sentences_with_embeddings(answer, chunks, co, threshold=0.9):
    """
    For each sentence, find the chunk with highest cosine similarity
    (above threshold), to highlight it.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    chunk_embeddings = embed_chunks(chunks, co)
    sentence_embeddings = embed_sentences(sentences, co)
    
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

def call_llama_model(model_name, prompt):
    # Named to match user dropdown
    model_map = {
        "llama3": "meta-llama/Llama-3.2-3B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
    }
    hf_model = model_map.get(model_name)
    if not hf_model:
        return "❌ Unsupported model selected."

    try:
        completion = client.chat.completions.create(
            model=hf_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message["content"].strip()
    except Exception as e:
        return f"❌ LLaMA API error: {e}"

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

    if gen_model_name in ["command", "command-light"]:
        cohere_model = gen_model_name
        response = co.generate(
            model=cohere_model,
            prompt=prompt,
            max_tokens=300,
            temperature=0.0
        )
        return response.generations[0].text.strip()

    elif gen_model_name in ["llama2", "llama3", "mistral"]:
        return call_llama_model(gen_model_name, prompt)

    else:
        return "❌ Unknown model selected."




def generate_llm_response_page(prev_page):
    st.title("Step 5: Final Response")

    top_k_results, user_query = get_llm_data()

    if not top_k_results or not user_query:
        st.warning("⚠️ No data found. Please run the similarity search first.")
        return

    # Select model
    model_name = st.selectbox("Choose Cohere Generation Model", [
    "command",
    "command-light",
    "llama3",
    "mistral"
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
            highlights = match_answer_sentences_with_embeddings(response, top_k_results, co, threshold=0.4)
        highlighted_html = render_highlighted_response(highlights)

        st.markdown("---")
        st.markdown("### ✨ LLM Response with Chunk Highlighting")
        st.markdown(highlighted_html, unsafe_allow_html=True)





    # Navigation buttons
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("⬅️ Back"):
            prev_page()
