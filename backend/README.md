# CodePilot AI - Backend

FastAPI backend for CodePilot AI with clean architecture.

## Project Structure

```
backend/
├── app/
│   ├── api/              # API endpoints and routes
│   ├── services/         # Business logic layer
│   ├── repository/       # Data access layer
│   ├── rag/              # RAG components (future)
│   ├── embeddings/       # Embeddings management (future)
│   ├── vectorstore/      # Vector store operations (future)
│   ├── prompts/          # Prompt templates (future)
│   ├── models/           # Data models and schemas
│   └── utils/            # Utility functions
├── main.py               # Application entry point
└── requirements.txt      # Python dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check endpoint
- `POST /repository/clone` - Clone a GitHub repository into `indexed_repos/<name>/source`
- `POST /repository/index` - Index an already-cloned repository. Returns structured
  statistics (files scanned, chunks, embeddings, vectors stored, per-repository
  vector count). Re-indexing replaces the repository's previous vectors, so stale
  chunks from deleted or shrunk files are removed.
- `POST /rag/query` - Ask a natural-language question about an indexed repository
  (history-backed: `app/api/rag.py`).

### RAG repository semantics

- Repository does not exist locally -> `HTTP 404`.
- Repository exists locally but has not been indexed -> `HTTP 404` (index it
  via `POST /repository/index` first).
- Repository is indexed but the question has no relevant context ->
  `HTTP 200` with `insufficient_context=true` and no fabricated
  repository-specific answer.
- `insufficient_context` is never conflated with a repository that does not
  exist.

## Development

The server runs on `http://localhost:8000` by default.

API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
