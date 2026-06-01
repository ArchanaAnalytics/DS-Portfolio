# 📄 LLM-Powered Document Assistant

A document-based Q&A chatbot built with **FastAPI** and a **local open-source LLM** (via Ollama). Ask questions about your own documents — no OpenAI API key required.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Local LLM** | Uses Ollama (Mistral, LLaMA3, Gemma, etc.) |
| **Async API** | FastAPI with async HTTP calls to reduce latency |
| **Prompt Optimization** | Structured prompts with document context injection |
| **Fallback Strategy** | Automatically retries once on LLM failure |
| **Response Caching** | TTL-based in-memory cache to skip redundant LLM calls |
| **Rate Limiting** | Per-IP sliding window limiter (10 req/min default) |
| **Clean Chat UI** | Minimal browser-based frontend, no React needed |

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

## 📌 Tech Stack

- Python 3.10+
- FastAPI
- Ollama (local LLM runner)
- httpx (async HTTP)
- Uvicorn (ASGI server)
