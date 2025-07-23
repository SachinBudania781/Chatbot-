# ingest.py
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_DIR = "pdfs"
CHROMA_DB_DIR = "vectorstore"

def load_and_index():
    docs = []
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(PDF_DIR, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vectordb.persist()
    print(f"✅ Vectorstore created at '{CHROMA_DB_DIR}'")

if __name__ == "__main__":
    load_and_index()
