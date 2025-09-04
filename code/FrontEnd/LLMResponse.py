import streamlit as st
import cohere

COHERE_API_KEY = "3exNVoOunyTRZVh7zbEpbrSZaLVlvYfdLEyvBgRp"
co = cohere.Client(COHERE_API_KEY)


def get_llm_data():
    llm_data = st.session_state.get("llm_data", {})
    return llm_data.get("top_k_results", []), llm_data.get("user_query", "")

def call_llm(gen_model_name, query, top_k_results, context):
    prompt = f"""
    You are a helpful AI assistant. Use the top-k similar results and any additional context to answer the user's query.

    User Query:
    {query}

    Top-K Retrieved Results:
    {'; '.join(top_k_results)}

    Additional Context:
    {context}

    Provide a helpful response:
    """

    response = co.generate(
        model=gen_model_name,  # or 'command-r-plus' if available
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )

    return response.generations[0].text.strip()


def generate_llm_response_page(prev_page):
    st.title("Step 5: Final Response")

    top_k_results, user_query = get_llm_data()

    if not top_k_results or not user_query:
        st.warning("⚠️ No data found. Please run the similarity search first.")
        return

    # Select model
    model_name = st.selectbox("Choose Cohere Generation Model", [
    "command",
    "command-light"
    ])

    # Context input
    context = st.text_area("Optional additional context for the LLM:", height=150)

    # Show Top-K results
    st.markdown("#### 🔍 Top-K Retrieved Results")
    for i, res in enumerate(top_k_results, 1):
        st.markdown(f"**{i}.** {res}")

    # Submit Button: generates LLM response
    generate_clicked = st.button("🚀 Generate LLM Response")

    response = None
    if generate_clicked:
        with st.spinner("Generating response..."):
            response = call_llm(model_name, user_query, top_k_results, context)

    # Always render this at the bottom AFTER generation
    if generate_clicked and response:
        st.markdown("---")
        st.markdown("### ✨ LLM Response")
        st.success(response)

    # Navigation buttons
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("⬅️ Back"):
            prev_page()
