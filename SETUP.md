# CodePilot AI - Setup Commands

Quick reference for setting up the project from scratch.

## Complete Setup (First Time)

### 1. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows (Git Bash):
source venv/Scripts/activate
# Windows (CMD):
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create .env file from example (optional)
cp .env.example .env

# Start the backend server
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup (New Terminal)
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

## Daily Development

### Start Backend
```bash
cd backend
source venv/Scripts/activate  # Windows Git Bash
uvicorn main:app --reload --port 8000
```

### Start Frontend
```bash
cd frontend
npm run dev
```

## Verification

1. Backend: Open `http://localhost:8000/docs` - Should show FastAPI Swagger UI
2. Frontend: Open `http://localhost:5173` - Should show CodePilot AI page
3. Integration: Click "Check Backend" button - Should display welcome message

## Troubleshooting

### Backend Issues
- **Port 8000 already in use**: Change port with `--port 8001`
- **Module not found**: Ensure venv is activated and dependencies installed
- **Python version**: Requires Python 3.8+

### Frontend Issues
- **Port 5173 already in use**: Vite will automatically use next available port
- **npm install fails**: Try deleting `node_modules` and `package-lock.json`, then reinstall
- **Backend connection fails**: Ensure backend is running on port 8000

### CORS Issues
- Backend already configured to allow `http://localhost:5173`
- If using different port, update CORS origins in `backend/main.py`

## Production Build

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm run build
npm run preview
```

## Environment Variables

### Backend (.env)
```env
HOST=0.0.0.0
PORT=8000
RELOAD=true
API_VERSION=v1
API_TITLE=CodePilot AI
```

Future environment variables (not yet needed):
- OpenAI/Anthropic API keys
- Vector database configuration
- Authentication secrets

## Dependencies Update

### Backend
```bash
pip install --upgrade -r requirements.txt
```

### Frontend
```bash
npm update
```

## Clean Install

### Backend
```bash
rm -rf venv
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

### Frontend
```bash
rm -rf node_modules package-lock.json
npm install
```
