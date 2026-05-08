# 📄 LLM-Powered Document Assistant

A document-based Q&A chatbot built with **FastAPI** and a **local open-source LLM** (via Ollama). Ask questions about your own documents — no OpenAI API key required.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Local LLM** | Uses Ollama (Mistral, LLaMA3, Gemma, etc.) — 100% free & private |
| **Async API** | FastAPI with async HTTP calls to reduce latency |
| **Prompt Optimization** | Structured prompts with document context injection |
| **Fallback Strategy** | Automatically retries once on LLM failure |
| **Response Caching** | TTL-based in-memory cache to skip redundant LLM calls |
| **Rate Limiting** | Per-IP sliding window limiter (10 req/min default) |
| **Clean Chat UI** | Minimal browser-based frontend, no React needed |

---

## 🗂️ Project Structure

```
doc-assistant/
│
├── app/
│   ├── main.py          # FastAPI app — routes, prompt builder, LLM caller
│   ├── cache.py         # TTL in-memory cache
│   └── limiter.py       # Per-IP rate limiter
│
├── docs/                # ← Drop your .txt documents here
│   └── sample.txt
│
├── frontend/
│   └── index.html       # Chat UI (served by FastAPI)
│
├── requirements.txt
├── .env                 # Config (model name, Ollama URL)
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install Ollama
Download from https://ollama.com and install it.

Then pull a model (Mistral recommended):
```bash
ollama pull mistral
```

### 2. Clone & install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/doc-assistant.git
cd doc-assistant
pip install -r requirements.txt
```

### 3. Add your documents
Place `.txt` files in the `docs/` folder.

### 4. Start Ollama (in a separate terminal)
```bash
ollama serve
```

### 5. Run the app (in VSCode terminal)
```bash
uvicorn app.main:app --reload
```

### 6. Open the chatbot
Go to: **http://localhost:8000**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ask` | Submit a question, get an answer |
| GET | `/health` | Check server + model status |
| GET | `/` | Serve the chat UI |

### Example request:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the CEO?"}'
```

### Example response:
```json
{
  "question": "Who is the CEO?",
  "answer": "The CEO is Sarah Mendes.",
  "cached": false,
  "response_time_ms": 1842.5
}
```

---

## 🧠 Architecture Decisions

- **Why Ollama?** It runs open-source models locally with a simple REST API — no API keys, no cost, full privacy.
- **Why async?** FastAPI's async support means the server doesn't block while waiting for the LLM, allowing multiple requests to be handled.
- **Why cache?** The same question asked twice shouldn't hit the LLM again. TTL ensures stale answers expire.
- **Why rate limiting?** Prevents the local LLM from being overloaded by too many rapid requests.

---

## 🔄 Swap the Model

Edit `.env`:
```
MODEL_NAME=llama3
```
Make sure you've pulled it first: `ollama pull llama3`

---

## 📌 Tech Stack

- Python 3.10+
- FastAPI
- Ollama (local LLM runner)
- httpx (async HTTP)
- Uvicorn (ASGI server)
