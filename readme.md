# PDF Question Answering Chatbot with Memory (LangChain + Streamlit)

This project is an interactive chatbot that allows users to ask questions based on the contents of a PDF document. It uses **LangChain** to build a Retrieval-Augmented Generation (RAG) pipeline with:

- PDF document loading and splitting
- Vector embeddings with OpenAI embeddings
- FAISS vector store for fast similarity search
- Conversational retrieval chain with memory to maintain chat context
- Custom prompt template for question answering
- Streamlit UI for a simple web-based chat interface

---

## Features

- Upload or load PDF documents and automatically chunk content
- Use embeddings and vector search for efficient retrieval
- Context-aware conversational Q&A with memory of previous interactions
- Simple, user-friendly Streamlit interface
- Ability to clear conversation and reset memory

---

