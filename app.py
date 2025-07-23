from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

PDF_DIR = "pdfs"
CHROMA_DB = "chroma_db"

def load_vectordb():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return Chroma(persist_directory=CHROMA_DB, embedding_function=embedding)

def get_chain():
    vectordb = load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain

qa_chain = get_chain()

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("frontend/index.html", "r") as f:
        return f.read()

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return JSONResponse(content={"error": "No question provided."}, status_code=400)
    result = qa_chain({"question": question})
    return JSONResponse(content={"answer": result["answer"]})
