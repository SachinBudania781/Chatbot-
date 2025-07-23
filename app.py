from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY", "sk-YOUR-REAL-KEY")
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    result = qa_chain({"question": user_input})
    return JSONResponse({"answer": result["answer"]})