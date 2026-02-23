import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# -------------------------
# CONFIG
# -------------------------

st.set_page_config(page_title="RAG Q&A App", layout="wide")
st.title("📄 RAG Question & Answer App")

groq_api_key = st.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# -------------------------
# Upload File
# -------------------------

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    st.success(f"Loaded {len(docs)} chunks from PDF")

    # Create Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Vector Store
    vector_store = Chroma.from_documents(
        docs,
        embedding,
        persist_directory="./chroma_db"
    )

    retriever = vector_store.as_retriever()

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG Chain
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question Input
    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke(question)

        st.subheader("Answer")
        st.write(response)
