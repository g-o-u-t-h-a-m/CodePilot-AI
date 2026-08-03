# CodePilot AI

An AI-powered code assistant built with clean architecture principles.

## 🏗️ Architecture

```
CodePilot-AI/
├── backend/          # FastAPI backend with modular architecture
├── frontend/         # React + Vite frontend
└── docs/             # Documentation
```

## 🚀 Quick Start

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000
```

Backend will be available at: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:5173`

## 📦 Project Structure

### Backend (Python + FastAPI)

```
backend/
├── app/
│   ├── api/              # API routes and endpoints
│   │   ├── health.py     # Health check endpoint
│   │   └── __init__.py
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

### Frontend (React + Vite)

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Home.jsx      # Home page component
│   │   └── Home.css      # Home page styles
│   ├── App.jsx           # Main app component
│   ├── App.css           # App styles
│   ├── main.jsx          # Entry point
│   └── index.css         # Global styles
├── index.html            # HTML template
├── vite.config.js        # Vite configuration
└── package.json          # Dependencies and scripts
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint - Welcome message |
| GET | `/health` | Health check - Service status |

## 🛠️ Technologies

**Backend:**
- FastAPI - Modern Python web framework
- Uvicorn - ASGI server
- Pydantic - Data validation

**Frontend:**
- React 18 - UI library
- Vite - Build tool and dev server
- Modern ES6+ JavaScript

## 📝 Development

### Backend Development

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The `--reload` flag enables hot-reloading during development.

### Frontend Development

```bash
cd frontend
npm run dev
```

Vite provides hot module replacement (HMR) for instant updates.

## 🧪 Testing the Integration

1. Start the backend server (port 8000)
2. Start the frontend dev server (port 5173)
3. Open `http://localhost:5173` in your browser
4. Click the "Check Backend" button
5. You should see: "Welcome to CodePilot AI! 🚀"

## 📚 Next Steps

This is the initial scaffolding with clean architecture. Future additions:

- **RAG System**: Implement retrieval-augmented generation
- **Embeddings**: Add code embedding generation
- **Vector Store**: Integrate vector database (ChromaDB/Pinecone)
- **Repository Indexing**: Index and analyze code repositories
- **AI Models**: Integrate LLM providers (OpenAI/Anthropic)
- **Authentication**: Add user authentication
- **Testing**: Unit and integration tests

## 📄 License

This project is private and not licensed for public use.

## 🤝 Contributing

This is a private project. Contribution guidelines will be added when the project becomes public.
