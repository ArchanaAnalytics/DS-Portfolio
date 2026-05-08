"""
LLM-Powered Document Assistant (RAG + FastAPI)

Features:
- FastAPI backend
- OpenAI LLM
- FAISS-based Retrieval (RAG)
- Async handling + retry
- Caching + rate limiting
- Source attribution
"""

import os
import time
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

# RAG components
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Custom modules
from app.cache import ResponseCache
from app.limiter import RateLimiter


# ── Load env ───────────────────────────────────────────────
load_dotenv()

app = FastAPI(title="Document Assistant API (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")


# ── Config ─────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "docs")

cache = ResponseCache()
limiter = RateLimiter(max_calls=10, period=60)


# ── LLM Setup ──────────────────────────────────────────────
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.3,
    api_key=OPENAI_API_KEY
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful document assistant.

Answer ONLY using the context below.
If the answer is not in the context, say:
"I couldn't find that in the provided documents."

Also include short references like [Source: filename].

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:
"""
)

qa_chain = prompt_template | llm | StrOutputParser()


# ── Load + Index Documents (RAG) ───────────────────────────
def load_and_index_documents():
    documents = []

    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        return None

    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_FOLDER, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()

            # Attach metadata
            for doc in docs:
                doc.metadata["source"] = filename

            documents.extend(docs)

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


vector_store = load_and_index_documents()

if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
else:
    retriever = None


# ── Utility: Normalize Query (Better Cache Hit) ─────────────
def normalize_query(q: str) -> str:
    return " ".join(q.lower().strip().split())


# ── Retrieve Context ───────────────────────────────────────
def get_context_and_sources(question: str):
    if not retriever:
        return "No documents available.", []

    docs = retriever.invoke(question)

    context = ""
    sources = set()

    for d in docs:
        context += d.page_content + "\n\n"
        if "source" in d.metadata:
            sources.add(d.metadata["source"])

    return context.strip(), list(sources)


# ── Async LLM Call ─────────────────────────────────────────
async def call_llm(question: str, context: str) -> str:
    for attempt in range(2):
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: qa_chain.invoke({
                    "context": context,
                    "question": question
                })
            )
            return response.strip()
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(1)
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM error: {str(e)}"
                )


# ── Request / Response Models ──────────────────────────────
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    cached: bool
    response_time_ms: float


# ── Main Endpoint ──────────────────────────────────────────
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: Request, body: QuestionRequest):
    client_ip = request.client.host

    if not limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    norm_q = normalize_query(question)

    start_time = time.time()

    # Cache check
    cached_answer = cache.get(norm_q)
    if cached_answer:
        return AnswerResponse(
            question=question,
            answer=cached_answer["answer"],
            sources=cached_answer["sources"],
            cached=True,
            response_time_ms=round((time.time() - start_time) * 1000, 2)
        )

    # RAG retrieval
    context, sources = get_context_and_sources(question)

    # LLM call
    answer = await call_llm(question, context)

    # Store cache
    cache.set(norm_q, {
        "answer": answer,
        "sources": sources
    })

    return AnswerResponse(
        question=question,
        answer=answer,
        sources=sources,
        cached=False,
        response_time_ms=round((time.time() - start_time) * 1000, 2)
    )


# ── Health Check ───────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "documents_loaded": retriever is not None,
        "cache_size": cache.size()
    }