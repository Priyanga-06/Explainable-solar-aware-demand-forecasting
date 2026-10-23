# Single-container deployment (backend + static frontend)
# Works on Render, Railway, Fly.io, or any Docker host

FROM python:3.11-slim

# Install Node.js for frontend build
RUN apt-get update && apt-get install -y nodejs npm

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install frontend dependencies and build
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Copy backend code
WORKDIR /app
COPY backend/ ./backend/
COPY config/ ./config/
COPY data/ ./data/
COPY .env.example .env

# Expose port (Render sets PORT env var)
ENV PORT=8000
EXPOSE 8000

# Start API server
CMD uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
