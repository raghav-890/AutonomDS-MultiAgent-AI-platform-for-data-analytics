# AutonomDS — Deployment Guide

## Local Development (No Docker)

### Prerequisites
- Python 3.11+
- Redis (for Celery async mode — optional)
- Ollama (for local LLM — optional, system works without it)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/autonomds.git
cd autonomds

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
make env              # Copies .env.example → .env
# Edit .env as needed

# 5. Create runtime directories
make dirs

# 6. (Optional) Pull Ollama model
ollama pull llama3.2   # or phi3 for lighter hardware

# 7. Start services
make api              # Terminal 1: FastAPI on :8000
make frontend         # Terminal 2: Streamlit on :8501
make worker           # Terminal 3: Celery worker (optional)
```

---

## Docker Compose (Recommended)

Runs the full stack: API + Frontend + Celery Worker + Redis + Flower monitoring.

```bash
# 1. Build images
make docker-build

# 2. Start all services
make docker-up

# Access:
# API:       http://localhost:8000
# API Docs:  http://localhost:8000/docs
# Frontend:  http://localhost:8501
# Flower:    http://localhost:5555  (Celery monitoring)
```

### Environment variables for Docker

Edit `.env` before running Docker:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434   # Use Docker service name
REDIS_URL=redis://redis:6379/0
```

---

## Free Cloud Deployment

### Option A: Render (Recommended for API)

1. Connect your GitHub repository to [Render](https://render.com)
2. Create a **Web Service** pointing to `docker/Dockerfile.api`
3. Set environment variables in Render dashboard
4. Uses the existing `deployment/render.yaml`

**Free tier:** 750 hours/month, spins down after inactivity.

### Option B: Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Deploy
railway up
```

Uses `railway.json` at the project root.

### Option C: Streamlit Cloud (Frontend Only)

1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point to `app/frontend/streamlit_app.py`
4. Set `API_BASE_URL` secret to your deployed API URL

**Note:** Set `API_BASE_URL` in Streamlit secrets so the frontend points to your deployed API.

### Option D: HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Push code or connect GitHub repo
4. Add secrets for environment variables

**HF Spaces `app.py`:** Copy `app/frontend/streamlit_app.py` to `app.py` at root.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `AUTONOMDS_ENV` | `development` | `development \| production \| test` |
| `LLM_PROVIDER` | `ollama` | `ollama \| huggingface` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name |
| `HUGGINGFACE_API_TOKEN` | `` | HF token (for fallback) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker URL |
| `UPLOAD_DIR` | `./uploads` | Dataset upload directory |
| `REPORTS_DIR` | `./reports` | Generated reports directory |
| `MAX_UPLOAD_MB` | `200` | File size limit |
| `N_CV_FOLDS` | `5` | Cross-validation folds |
| `OPTUNA_TRIALS` | `20` | HPO trials per model |
| `SECRET_KEY` | *(required)* | App secret key |

---

## Monitoring

### MLflow UI

```bash
make mlflow-ui
# Open http://localhost:5000
```

### Celery Flower

```bash
# With Docker
# Open http://localhost:5555

# Without Docker
celery -A app.api.main.celery_app flower --port=5555
```

### Health Check

```bash
curl http://localhost:8000/health
# → {"status": "healthy", "version": "1.0.0", ...}
```

---

## Production Hardening Checklist

- [ ] Set `AUTONOMDS_ENV=production` 
- [ ] Set a strong `SECRET_KEY`
- [ ] Disable `/docs` (set in `is_production` check)
- [ ] Configure proper CORS origins
- [ ] Use Redis Sentinel or Redis Cloud for HA
- [ ] Set `MAX_UPLOAD_MB` to a reasonable limit
- [ ] Configure log aggregation (Loki/Datadog/CloudWatch)
- [ ] Set up Prometheus + Grafana for metrics (optional)
- [ ] Enable rate limiting (add slowapi middleware)
