# CodePilot AI - Frontend

React + Vite frontend for CodePilot AI.

## Project Structure

```
frontend/
├── src/
│   ├── pages/           # Page components
│   │   ├── Home.jsx     # Home page
│   │   └── Home.css     # Home page styles
│   ├── App.jsx          # Main app component
│   ├── App.css          # App styles
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── index.html           # HTML template
├── vite.config.js       # Vite configuration
└── package.json         # Dependencies and scripts
```

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The app will open at `http://localhost:5173`.

## Features

- **Home Page**: Displays the CodePilot AI title with a button to check backend connectivity
- **Backend Integration**: Connects to the FastAPI backend at `http://localhost:8000`

## Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Backend Connection

The frontend communicates with the backend API at `http://localhost:8000`. Make sure the backend is running before testing the "Check Backend" button.
