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

## Development

The server runs on `http://localhost:8000` by default.

API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
