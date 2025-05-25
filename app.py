import streamlit as st
from llm import initialize_chain
import tempfile
import os
from streamlit_chat import message

# Page configuration
st.set_page_config(page_title="PDF Q&A Chatbot", layout="centered")
st.title("🤖 Chat with your PDF")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_pdf_path = tmp.name

    st.success("✅ PDF uploaded successfully!")

    # Initialize chain and memory
    chain, memory = initialize_chain(temp_pdf_path)

    # Chat history state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat input
    user_question = st.chat_input("Ask something about the PDF...")

    if user_question:
        # Get response from LLM
        result = chain({"question": user_question})
        answer = result["answer"]
        st.session_state.chat_history.append((user_question, answer))

    # Display chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        message(q, is_user=True, key=f"user_{i}")
        message(a, is_user=False, key=f"bot_{i}")

else:
    st.info("📄 Please upload a PDF file to begin chatting.")
