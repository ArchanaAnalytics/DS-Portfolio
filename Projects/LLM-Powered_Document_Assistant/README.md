# 📄 RAG Document Assistant with FastAPI, OpenAI & FAISS

A document-based Q&A chatbot built with **FastAPI**, **OpenAI**, and **FAISS-powered Retrieval-Augmented Generation (RAG)**. Ask questions about your own documents and receive accurate answers grounded in retrieved context, complete with source attribution.

---

## 🚀 Features

| Feature                         | Details                                                      |
| ------------------------------- | ------------------------------------------------------------ |
| **OpenAI LLM**                  | Uses OpenAI models (GPT-4o Mini) via LangChain    |
| **RAG Pipeline**                | Retrieves relevant document chunks using FAISS vector search |
| **OpenAI Embeddings**           | Generates semantic embeddings for document retrieval         |
| **Async API**                   | FastAPI with asynchronous request handling                   |
| **Prompt Optimization**         | Structured prompts with retrieved document context           |
| **Source Attribution**          | Returns document filenames used to generate answers          |
| **Fallback Strategy**           | Automatically retries once on LLM failure                    |
| **Response Caching**            | TTL-based in-memory cache to avoid redundant LLM calls       |
| **Rate Limiting**               | Per-IP sliding window limiter (10 req/min default)           |
| **Automatic Document Indexing** | Loads and indexes `.txt` files on startup                    |
| **Clean Chat UI**               | Lightweight browser-based frontend, no React required        |

---

## 🏗️ Architecture

```text
Documents
   │
   ▼
Chunking
   │
   ▼
OpenAI Embeddings
   │
   ▼
FAISS Vector Store
   │
   ▼
Retriever
   │
   ▼
GPT-4o Mini
   │
   ▼
Answer + Sources
```

---

## 📂 Project Structure

```text
project/
│
├── app/
│   ├── cache.py
│   ├── limiter.py
│
├── docs/
│
├── frontend/
│   ├── index.html
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

## 📋 Requirements

* Python 3.10+
* OpenAI API Key

---

## ⚙️ Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4o-mini
DOCS_FOLDER=docs
```

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/document-assistant.git

cd document-assistant

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📂 Document Setup

Place your `.txt` documents inside the `docs/` directory:

```text
docs/
├── company_overview.txt
├── handbook.txt
└── policies.txt
```

On startup, the application:

1. Loads all `.txt` files from the documents folder.
2. Splits documents into chunks.
3. Generates embeddings using OpenAI Embeddings.
4. Creates a FAISS vector index.
5. Uses semantic search to retrieve relevant content for each query.

---

## ▶️ Run the Application

```bash
uvicorn main:app --reload
```

The application will be available at:

```text
http://localhost:8000
```

---

## 🔌 API Endpoints

| Method | Endpoint  | Description                                  |
| ------ | --------- | -------------------------------------------- |
| POST   | `/ask`    | Submit a question and receive an answer      |
| GET    | `/health` | Check API, model, cache, and document status |
| GET    | `/`       | Serve the chat UI                            |

---

## 📥 Example Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Who is the CEO?"}'
```

---

## 📤 Example Response

```json
{
  "question": "Who is the CEO?",
  "answer": "The CEO is Sarah Mendes.",
  "sources": ["company_overview.txt"],
  "cached": false,
  "response_time_ms": 1324.71
}
```

---

## ❤️ Health Check

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model": "gpt-4o-mini",
  "documents_loaded": true,
  "cache_size": 3
}
```

---

## 🔍 Retrieval-Augmented Generation (RAG)

The assistant does not rely solely on the LLM's training data.

For each question:

1. Relevant document chunks are retrieved from FAISS.
2. Retrieved context is injected into the prompt.
3. GPT generates an answer using only the provided context.
4. Source filenames are returned alongside the response.

If the answer cannot be found in the documents, the assistant responds:

> "I couldn't find that in the provided documents."

---

## 🛠️ Tech Stack

* Python 3.10+
* FastAPI
* LangChain
* OpenAI GPT Models
* OpenAI Embeddings
* FAISS
* Uvicorn
* httpx
* Pydantic
* python-dotenv

---

## ⭐ Key Features

* Retrieval-Augmented Generation (RAG)
* Semantic Search using FAISS
* OpenAI Embeddings
* Source Attribution
* Async Request Handling
* Retry Mechanism
* Response Caching
* Rate Limiting
* FastAPI REST API

---
