from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import tempfile
import os


def initialize_chain(pdf_path):
    # Load PDF and split
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(data)

    # Embedding and vectorstore
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embedding)
    retriever = vectorstore.as_retriever()

    # Prompt template
    template = """
    You are a helpful assistant that can answer questions about the following text:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # LLM
    llm = ChatGroq(model="llama3-8b-8192")

    # Conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain, memory
