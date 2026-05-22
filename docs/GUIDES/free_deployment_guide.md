# AutonomDS — Free Deployment Guide

## Realistic Architecture for Free Cloud

> [!IMPORTANT]
> **Ollama cannot run on free cloud** — it needs 4–8GB RAM minimum and no free tier provides that. On the cloud, the system automatically uses the **HuggingFace Inference API** as the LLM (already built in as a fallback). This is completely free with a HuggingFace account.

### Free Stack You'll Use

| Component | Service | Free Tier Limits |
|---|---|---|
| 🖥️ **API (FastAPI)** | [Render](https://render.com) | 512 MB RAM, spins down after 15 min idle |
| 🎨 **Frontend (Streamlit)** | [Streamlit Community Cloud](https://streamlit.io/cloud) | 1 GB RAM, always on |
| 📦 **Redis (Task Queue)** | [Upstash](https://upstash.com) | 10,000 commands/day, 256 MB |
| 🤖 **LLM** | [HuggingFace Inference API](https://huggingface.co/settings/tokens) | ~1,000 req/day free |
| 🧬 **Code Repository** | [GitHub](https://github.com) | Unlimited public repos |
| ⚙️ **CI/CD** | GitHub Actions | 2,000 min/month free |

> [!NOTE]
> **ChromaDB** runs *embedded inside the API* — no separate service needed.  
> **SQLite** runs *embedded inside the API* — no separate DB needed.  
> **MLflow** uses SQLite — runs embedded too.

---

## Step 0 — One-Time Code Changes Required

Before deploying, two small changes are needed to make the system cloud-compatible.

### Change 1: Reduce default Optuna trials for free tier RAM

Render's free tier has only 512 MB RAM. 30 Optuna trials is too heavy.

Open `.env.example` — when you set env vars on Render, use these values:
```
OPTUNA_N_TRIALS=10
CROSS_VAL_FOLDS=3
MAX_UPLOAD_SIZE_MB=50
```

### Change 2: Add a Streamlit `secrets.toml` file for the frontend

Create this file at:
```
app/frontend/.streamlit/secrets.toml
```

With this content:
```toml
API_BASE_URL = "https://your-render-app-name.onrender.com"
```

> You'll fill in the real Render URL after deployment (Step 4).

### Change 3: Update `streamlit_app.py` to read API URL from secrets

Open `app/frontend/streamlit_app.py` and find where `API_BASE_URL` or `streamlit_api_url` is set. Change it to:
```python
import streamlit as st
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
```

---

## Step 1 — Push to GitHub

```bash
cd autonomous-ds-agent

# Initialise git if not already done
git init
git add .
git commit -m "Initial commit: AutonomDS v1.0.0"

# Create a repo on GitHub (github.com → New repository)
# Then connect and push:
git remote add origin https://github.com/YOUR_USERNAME/autonomds.git
git branch -M main
git push -u origin main
```

> [!CAUTION]
> Make sure `.gitignore` includes `.env`, `uploads/`, `experiments/`, `chroma_db/` — these must NOT be pushed to GitHub. The existing `.gitignore` already covers these.

---

## Step 2 — Get a Free HuggingFace API Token (LLM)

1. Go to [huggingface.co](https://huggingface.co) → Sign up (free)
2. Go to **Settings → Access Tokens → New Token**
3. Choose **Read** permission, name it `autonomds`
4. Copy the token — it looks like `hf_xxxxxxxxxxxxxxxxx`

You'll use this in Step 3 and Step 4 as `HUGGINGFACE_API_TOKEN`.

---

## Step 3 — Set Up Free Redis on Upstash

1. Go to [upstash.com](https://upstash.com) → Sign up (free)
2. Click **Create Database → Redis**
3. Choose a name: `autonomds-redis`
4. Region: closest to you (e.g., US-East-1)
5. After creation, go to the database → **Details tab**
6. Copy the **Redis URL** — it looks like:
   ```
   rediss://default:xxxxxxxxxxxxxxxx@us1-xxx.upstash.io:6380
   ```

You'll use this as `REDIS_URL`, `CELERY_BROKER_URL`, and `CELERY_RESULT_BACKEND` in Step 4.

---

## Step 4 — Deploy API on Render (Free)

1. Go to [render.com](https://render.com) → Sign up with GitHub (free)
2. Click **New → Web Service**
3. Connect your GitHub repo `autonomds`
4. Configure:

| Setting | Value |
|---|---|
| **Name** | `autonomds-api` |
| **Region** | Oregon (US West) |
| **Branch** | `main` |
| **Runtime** | **Docker** |
| **Dockerfile Path** | `docker/Dockerfile.api` |
| **Instance Type** | **Free** |

5. Click **Advanced → Add Environment Variables** and add ALL of these:

```
APP_ENV=production
APP_VERSION=1.0.0
DEBUG=false
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">

API_HOST=0.0.0.0
API_PORT=10000
API_WORKERS=1

# LLM — HuggingFace (Ollama won't work on cloud)
LLM_PROVIDER=huggingface
HUGGINGFACE_API_TOKEN=hf_your_token_here
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Embeddings (downloaded at startup, ~80MB)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Redis (from Upstash)
REDIS_URL=rediss://default:xxx@us1-xxx.upstash.io:6380
CELERY_BROKER_URL=rediss://default:xxx@us1-xxx.upstash.io:6380
CELERY_RESULT_BACKEND=rediss://default:xxx@us1-xxx.upstash.io:6380

# Storage (ephemeral on free Render — files lost on restart)
UPLOAD_DIR=/tmp/autonomds/uploads
REPORTS_DIR=/tmp/autonomds/reports
CHROMA_PERSIST_DIR=/tmp/autonomds/chroma_db

# MLflow (SQLite embedded)
MLFLOW_TRACKING_URI=sqlite:////tmp/autonomds/mlflow.db

# Training (reduced for free tier RAM)
OPTUNA_N_TRIALS=10
CROSS_VAL_FOLDS=3
MAX_UPLOAD_SIZE_MB=50
TRAINING_DEVICE=cpu

# CORS — update after deploying Streamlit
ALLOWED_ORIGINS=https://your-app.streamlit.app,http://localhost:8501

LOG_LEVEL=INFO
LOG_FORMAT=json
```

6. Click **Create Web Service**
7. Wait ~5-10 minutes for the Docker build
8. Once deployed, copy your URL: `https://autonomds-api-xxxx.onrender.com`

> [!WARNING]
> **Free Render tier sleeps after 15 minutes of inactivity.** The first request after sleep takes ~30 seconds to wake up. This is normal for the free tier. Use Render's "Cron Job" ping (or UptimeRobot free) to keep it alive.

---

## Step 5 — Deploy Frontend on Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub
2. Click **New app**
3. Configure:

| Setting | Value |
|---|---|
| **Repository** | `YOUR_USERNAME/autonomds` |
| **Branch** | `main` |
| **Main file path** | `app/frontend/streamlit_app.py` |
| **App URL** | `autonomds` (becomes: `autonomds.streamlit.app`) |

4. Click **Advanced settings → Secrets** and add:
```toml
API_BASE_URL = "https://autonomds-api-xxxx.onrender.com"
```

5. Click **Deploy**

> [!NOTE]
> Streamlit Cloud automatically installs from `requirements.txt`. First deployment takes ~3-5 minutes.

---

## Step 6 — Update CORS on Render

Once you have your Streamlit URL (e.g., `https://autonomds.streamlit.app`):

1. Go to Render → your service → **Environment**
2. Update `ALLOWED_ORIGINS`:
   ```
   ALLOWED_ORIGINS=https://autonomds.streamlit.app,http://localhost:8501
   ```
3. Render will auto-redeploy

---

## Step 7 — Verify Everything Works

```bash
# 1. Check API health
curl https://autonomds-api-xxxx.onrender.com/health

# Expected:
# {"status": "healthy", "version": "1.0.0", ...}

# 2. Check API docs
# Open: https://autonomds-api-xxxx.onrender.com/docs

# 3. Open Streamlit frontend
# Open: https://autonomds.streamlit.app
```

---

## Free Tier Limitations to Know

| Limitation | Impact | Workaround |
|---|---|---|
| Render sleeps after 15 min | First request is slow | UptimeRobot ping every 10 min (free) |
| No Ollama on cloud | LLM uses HuggingFace API | Already built in as fallback |
| HF free: ~1,000 req/day | LLM quota | Use smaller model or cache responses |
| Render 512 MB RAM | Large datasets may crash | Limit uploads to 50 MB |
| Ephemeral disk on Render | Uploads lost on restart | Use Cloudflare R2 or Supabase Storage for persistence (both free) |
| Upstash 10k cmd/day | Celery task tracking | Sufficient for light use |
| Streamlit 1 GB RAM | ML training may be slow | Training happens on the API, not frontend |

---

## Optional: Keep Render API Awake (Free)

Use [UptimeRobot](https://uptimerobot.com) (free tier: 50 monitors):

1. Sign up at uptimerobot.com
2. **New Monitor → HTTP(s)**
3. URL: `https://autonomds-api-xxxx.onrender.com/health`
4. Monitoring interval: **10 minutes**

This sends a ping every 10 minutes so Render never sleeps.

---

## Optional: Persistent File Storage with Supabase (Free)

Render's free tier has **ephemeral disk** — uploaded files are wiped on restart. For persistence:

1. Sign up at [supabase.com](https://supabase.com) (free: 500 MB storage)
2. Create a project → **Storage → New bucket** → `autonomds-uploads`
3. Install: `pip install supabase`
4. Modify `app/api/routes/upload.py` to save to Supabase Storage instead of local disk

This is optional — the system works without it, but uploads won't survive Render restarts.

---

## Full Deployment Checklist

- [ ] Push code to GitHub (with `.env` in `.gitignore`)
- [ ] Get HuggingFace API token
- [ ] Create Upstash Redis database, copy URL
- [ ] Deploy API on Render (Docker), set all env vars
- [ ] Wait for Render build to complete, copy URL
- [ ] Create `app/frontend/.streamlit/secrets.toml` with Render URL
- [ ] Push secrets file update to GitHub
- [ ] Deploy frontend on Streamlit Cloud
- [ ] Update `ALLOWED_ORIGINS` on Render with Streamlit URL
- [ ] Test: `curl your-render-url/health`
- [ ] Test: Upload a CSV from the Streamlit UI
- [ ] (Optional) Set up UptimeRobot to prevent Render sleep
- [ ] (Optional) Set up Supabase Storage for persistent uploads

---

## Architecture Diagram (Free Cloud)

```
User Browser
     │
     ▼
Streamlit Cloud ──────────────────────────────────────┐
(share.streamlit.app)                                  │
  API_BASE_URL = Render URL                            │
     │                                                 │
     │  HTTP REST calls                                │
     ▼                                                 │
Render (Free Tier)  ◄─────────────────────────────────┘
FastAPI + Celery Worker                        
  LLM: HuggingFace Inference API ──────► HuggingFace
  Queue: Upstash Redis            ──────► Upstash
  DB: SQLite (embedded, ephemeral)
  VectorDB: ChromaDB (embedded, ephemeral)
  Storage: /tmp/ (ephemeral)
```

---

## Cost Summary

| Service | Cost |
|---|---|
| Render (Web Service) | **$0/month** |
| Streamlit Cloud | **$0/month** |
| Upstash Redis | **$0/month** |
| HuggingFace API | **$0/month** |
| GitHub | **$0/month** |
| UptimeRobot | **$0/month** |
| **Total** | **$0/month** ✅ |
