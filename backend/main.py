import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, repository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodePilot AI",
    description="AI-powered code assistant with RAG capabilities",
    version="0.1.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(repository.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to CodePilot AI! 🚀",
        "status": "online",
        "version": "0.1.0"
    }
