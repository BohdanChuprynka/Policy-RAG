# Railway Deployment Guide

## Architecture on Railway

You'll create **3 services** in a single Railway project:

```
Railway Project
├── Backend service   → Dockerfile.backend
├── Frontend service  → Dockerfile.frontend
└── Redis plugin      → managed by Railway (no Dockerfile)
```

---

## Step 1: Create a Railway Project

1. Go to [railway.app](https://railway.app) and create a **New Project**
2. Choose **"Deploy from GitHub repo"** and connect this repository

---

## Step 2: Add Redis

1. Inside your Railway project, click **"+ New"** → **"Database"** → **"Add Redis"**
2. Railway will provision a managed Redis instance and expose the connection string as `REDIS_URL`
3. That's it — no Dockerfile, no config. Railway handles persistence, health checks, and restarts

---

## Step 3: Deploy the Backend

1. Click **"+ New"** → **"GitHub Repo"** → select this repo again (or use "Empty Service" → connect repo)
2. Go to the service **Settings**:
   - **Root Directory**: leave as `/` (the Dockerfile references paths from repo root)
   - **Dockerfile Path**: `Railway/Dockerfile.backend`
3. Go to the **Variables** tab and add:
   - `OPENAI_API_KEY` = your OpenAI key
   - `REDIS_URL` = click **"Add Reference"** → select the Redis plugin's `REDIS_URL`
   - `ALLOW_PDF_INGEST` = `true` (if you want PDF upload enabled)
   - Any other env vars from your `.env` as needed
4. Railway auto-detects the `PORT` variable — **do NOT set PORT manually**
5. Click **Deploy** (or it auto-deploys on push)

### Generate a public domain for the backend

- Go to **Settings** → **Networking** → **"Generate Domain"**
- Note this URL (e.g., `https://your-backend-production.up.railway.app`)

---

## Step 4: Deploy the Frontend (Streamlit)

1. Click **"+ New"** → **"GitHub Repo"** → select this repo again
2. Go to **Settings**:
   - **Root Directory**: leave as `/`
   - **Dockerfile Path**: `Railway/Dockerfile.frontend`
3. Go to **Variables** and add:
   - `API_BASE_URL` = the backend's Railway domain from Step 3
     (e.g., `https://your-backend-production.up.railway.app`)
   - `OPENAI_API_KEY` = your OpenAI key
4. Generate a public domain for the frontend the same way

---

## Step 5: Link Redis to Backend

Railway makes cross-service references easy:

1. Go to the **Backend service** → **Variables**
2. Click **"Add Reference"** → choose the Redis service
3. Select `REDIS_URL` — Railway will inject the internal connection string
   (format: `redis://default:password@redis.railway.internal:6379`)

Using the **internal** hostname (`*.railway.internal`) means traffic stays inside
Railway's private network — no latency or egress costs.

---

## Environment Variables Summary

### Backend service
| Variable           | Source                          |
|--------------------|---------------------------------|
| `PORT`             | Auto-set by Railway (don't set) |
| `REDIS_URL`        | Reference from Redis plugin     |
| `OPENAI_API_KEY`   | Manual                          |
| `ALLOW_PDF_INGEST` | Manual (optional, default false)|

### Frontend service
| Variable       | Source                               |
|----------------|--------------------------------------|
| `PORT`          | Auto-set by Railway (don't set)     |
| `API_BASE_URL` | Backend's Railway domain             |
| `OPENAI_API_KEY`| Manual                              |

---

## Troubleshooting

- **"Address already in use"**: You hardcoded a port. Always use `$PORT`.
- **Frontend can't reach backend**: Make sure `API_BASE_URL` uses `https://` and the full Railway domain.
- **Redis connection refused**: Verify `REDIS_URL` is a reference variable (not copy-pasted), so it stays in sync if Railway rotates credentials.
- **Build fails**: Railway builds from the repo root. Ensure `Railway/Dockerfile.backend` paths like `COPY src ./src` are correct relative to the repo root (they are — Railway sets the build context to the repo root).
