# Web-AI Production Deployment Plan

## Project Overview

A multi-tenant SaaS platform that lets businesses create AI-powered chat agents trained on their website content. Agents answer customer questions via web widget, LINE, and Instagram. Targets Japanese market.

### Tech Stack

- **Backend**: FastAPI (Python 3.11), Celery workers for background jobs (crawling, indexing, extraction)
- **Frontend**: React + Vite dashboard (admin panel), embeddable JS chat widget
- **Database**: PostgreSQL 16 (psycopg driver, raw SQL — no ORM)
- **Cache/Broker**: Redis (Celery task broker + analytics dirty-set)
- **AI**: Google Vertex AI — Gemini models for LLM, RAG corpus for retrieval-augmented generation
- **Storage**: Google Cloud Storage (crawled markdown, PDFs, extracted assets)
- **Auth**: Auth0 (JWT-based, org-scoped multi-tenancy)
- **Integrations**: LINE Messaging API, Instagram Graph API, SMTP email
- **Crawling**: crawl4ai + Playwright (headless Chromium browser crawling)

### Current State

- Runs locally via `docker-compose.yml` with 5 services: backend, celery-worker, dashboard (Nginx), postgres, redis
- Already has: Dockerfiles, health check endpoints (`/live`, `/health` with DB check), dynamic CORS middleware (per-bot origin validation), rate limiting (60 req/min per bot)
- All AI services already on GCP (Vertex AI, GCS) — project and credentials exist
- Missing: database migration tool (no Alembic), production env config, CI/CD pipeline

### Key Backend Environment Variables

```
DATABASE_URL, CELERY_BROKER_URL
GOOGLE_APPLICATION_CREDENTIALS, GCS_BUCKET, PROJECT_ID, LOCATION
VERTEX_RAG_MODEL, VERTEX_RAG_ENABLE_THINKING, VERTEX_RAG_THINK_BUDGET, VERTEX_RAG_MAX_OUTPUT_TOKENS
AUTH_ISSUER, AUTH_AUDIENCE, AUTH_JWKS_URL, SUPER_ADMIN_EMAILS
PUBLIC_BASE_URL, DASHBOARD_URL
LINE_CHANNEL_ID, LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN
INSTAGRAM_APP_ID, INSTAGRAM_APP_SECRET, INSTAGRAM_REDIRECT_URI, INSTAGRAM_WEBHOOK_VERIFY_TOKEN
SMTP_HOST, SMTP_PORT, SMTP_USE_TLS, SMTP_FROM_EMAIL, SMTP_USER, SMTP_PASSWORD
OPENAI_API_KEY, GOOGLE_API_KEY, ADMIN_API_KEY
```

### Key Dashboard Environment Variables

```
VITE_API_BASE_URL, VITE_AUTH0_DOMAIN, VITE_AUTH0_CLIENT_ID, VITE_AUTH0_AUDIENCE, VITE_AUTH_ORG_CLAIM, VITE_SUPER_ADMIN_EMAILS
```

---

## Target Production Architecture

```
                         Cloud Storage + Cloud CDN
                         (Dashboard static files)
                         app.yourdomain.com
                                   │
Users (Japan) ── HTTPS ────────────┤
                                   │
                              Cloud Run
                           (FastAPI API server)
                          asia-northeast1 (Tokyo)
                          min-instances: 0, max: 10
                          api.yourdomain.com
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                    │
         Neon Postgres       Vertex AI + GCS      Upstash Redis
        (asia-northeast1)    (already exists)      (Tokyo region)
         serverless,          Gemini models,        serverless,
         scale-to-zero        RAG corpus,           pay-per-command
                              Cloud Storage
                                   │
                              GCE VM (e2-micro)
                              asia-northeast1-a
                              Celery worker only
                              (~$8/month)
```

### Infrastructure Services

| Service | Provider | Purpose | Cost |
|---------|----------|---------|------|
| API server | GCP Cloud Run | FastAPI backend, auto-scales 0→10 | ~$0-5/month (pay per request) |
| Database | Neon (neon.tech) | Serverless PostgreSQL, Tokyo region | Free tier → $19/month |
| Redis broker | Upstash (upstash.com) | Celery task broker, serverless | Free tier → pay per command |
| Background workers | GCE VM (e2-micro) | Celery workers for crawling, indexing, extraction | ~$8/month |
| Dashboard hosting | GCP Cloud Storage + CDN | Static React build, edge-cached | ~$1-2/month |
| AI/ML | GCP Vertex AI | Gemini LLM + RAG retrieval | Usage-based (already set up) |
| File storage | GCP Cloud Storage | Crawled content, PDFs, images | Usage-based (already set up) |
| Auth | Auth0 | JWT authentication, org management | Free tier up to 7,500 MAU |
| Domain + SSL | Google-managed cert | Custom domain, auto-renewing SSL | Free |
| **Total base** | | | **~$10-15/month with no traffic** |

### Why These Choices

1. **Cloud Run over K8s/AWS**: Scales to zero ($0 idle), no cluster management, Docker-native, Tokyo region
2. **Neon over Cloud SQL**: Serverless Postgres, scales to zero, free tier, ~60% cheaper at low traffic ($9 vs $25+ for Cloud SQL)
3. **Upstash over Memorystore**: Serverless Redis, pay-per-command vs $30/month minimum, free tier
4. **GCE VM for Celery**: Celery needs persistent process (can't scale to zero), e2-micro is cheapest always-on option
5. **Stay on GCP**: Already using Vertex AI + GCS — no cross-cloud latency or egress fees
6. **Separate API/Worker images**: API image is slim (fast cold starts ~2-3s), worker image is heavy (Playwright + Chrome)

---

## Database Schema

All tables (dependency order — children before parents for deletion):

```
instagram_user_sessions, line_user_sessions,
instagram_channels, line_channels, line_design_configs,
availability_jobs, booking_link_jobs, bot_suggested_message_packs,
topic_jobs, index_jobs, discovery_jobs, asset_extraction_jobs,
conversation_messages, conversation_escalations, conversation_feedback, conversation_sessions,
topic_question_mappings, bot_extracted_topics,
bot_usage_daily, bot_sources_daily, bot_topics_daily,
bot_assets, bot_sources, bot_domains, bot_corpora,
rollup_watermarks, bots
```

No ORM is used. All queries are raw SQL via psycopg. Table creation is currently manual — needs Alembic migration setup.

---

## How the System Works

### Two Processing Paths

**User Chat (real-time, no queue):**
```
User message → FastAPI endpoint → Vertex AI RAG retrieval + Gemini LLM → Response
```
- Web widget uses streaming endpoint (`/v1/pk/{key}/chat/stream`) — tokens sent as NDJSON
- LINE uses blocking endpoint — full response generated, then sent via LINE push message API
- Instagram uses blocking endpoint — full response sent via Instagram Graph API
- All chat requests run in parallel via `asyncio.to_thread()` — no event loop blocking
- Rate limited: 60 requests per 60-second window per bot

**Background Jobs (Celery queue):**
```
Job trigger → Redis queue → Celery worker picks up → Process → Chain dependent jobs
```
- Single default queue, 10 concurrent workers (configurable)
- Job chain after crawl: crawl → topic extraction → asset extraction → menu extraction → booking link detection → prompt generation
- Time limits: crawl 10min, menu extraction 600s, asset extraction 300s, discovery 3min

### Key API Endpoints

**Public (widget/channel):**
- `POST /v1/pk/{key}/chat` — Web chat (non-streaming)
- `POST /v1/pk/{key}/chat/stream` — Web chat (streaming, NDJSON)
- `POST /webhooks/line/{bot_id}` — LINE webhook (signature-verified)
- `POST /webhooks/instagram/{bot_id}` — Instagram webhook

**Authenticated (dashboard):**
- `GET /v1/org/bots` — List bots for org
- `POST /v1/org/bots` — Create bot
- `DELETE /v1/org/bots/{bot_id}` — Delete bot (DB + GCS + RAG corpus cleanup)
- `POST /v1/org/bots/{bot_id}/test-chat` — Dashboard test chat
- `POST /v1/org/bots/{bot_id}/index/start` — Start crawling/indexing
- Various CRUD endpoints for bot config, LINE channels, Instagram channels, etc.

**Health:**
- `GET /live` — Always returns OK
- `GET /health` — Checks DB connectivity

### CORS

Dynamic CORS middleware extracts bot publishable key from URL path, validates request origin against verified hosts stored in database per bot. Dashboard origin needs explicit allowlisting.

---

## Codebase Changes Required Before Deploy

### 1. Database Migrations (Alembic)

No migration system exists. Need to set up Alembic for versioned schema changes.

- Add `alembic` to `backend/requirements.txt`
- Run `alembic init migrations` in `backend/`
- Configure `alembic.ini` to read `DATABASE_URL` from env
- Create initial migration capturing all ~27 existing tables
- Future schema changes via `alembic revision --autogenerate`
- Run `alembic upgrade head` on each deploy

### 2. Production Dockerfiles

**API Dockerfile (`Dockerfile.api`):**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
# No Playwright needed — API only serves HTTP requests
# Cloud Run sets PORT env var (default 8080)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

**Worker Dockerfile (`Dockerfile.worker`):**
```dockerfile
FROM python:3.11
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium --with-deps
COPY backend/ .
# Full image with Playwright + Chrome for web crawling
CMD ["celery", "-A", "infrastructure.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
```

**Dashboard:** Already has Nginx Dockerfile. For production, just `npm run build` and upload `dist/` to Cloud Storage.

### 3. Production Environment Config

Create separate env files for production with:
- Neon PostgreSQL connection string (with `?sslmode=require`)
- Upstash Redis connection string (with TLS: `rediss://`)
- Production Auth0 tenant credentials
- Production LINE/Instagram app credentials
- Production domain URLs (`PUBLIC_BASE_URL`, `DASHBOARD_URL`)
- GCP service account key or workload identity

### 4. CORS Update

Add production dashboard domain to allowed origins in CORS middleware.

### 5. Uvicorn Port

Cloud Run sets `PORT` env var. Ensure uvicorn reads it:
```python
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

---

## Deployment Steps

### Step 1: Create External Services

```bash
# Neon: Create project at neon.tech
# - Region: asia-northeast1 (Tokyo)
# - Create database "webai"
# - Copy connection string

# Upstash: Create Redis at upstash.com
# - Region: ap-northeast-1 (Tokyo)
# - Enable TLS
# - Copy connection string (rediss:// with TLS)

# GCP: Enable required APIs
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  cloudbuild.googleapis.com
```

### Step 2: Artifact Registry

```bash
gcloud artifacts repositories create webai \
  --repository-format=docker \
  --location=asia-northeast1
```

### Step 3: Build & Push Docker Images

```bash
# API image (slim, no Playwright)
gcloud builds submit \
  --tag asia-northeast1-docker.pkg.dev/PROJECT_ID/webai/api \
  --dockerfile=Dockerfile.api .

# Worker image (full, with Playwright + Chrome)
gcloud builds submit \
  --tag asia-northeast1-docker.pkg.dev/PROJECT_ID/webai/worker \
  --dockerfile=Dockerfile.worker .
```

### Step 4: Deploy API to Cloud Run

```bash
gcloud run deploy webai-api \
  --image=asia-northeast1-docker.pkg.dev/PROJECT_ID/webai/api \
  --region=asia-northeast1 \
  --platform=managed \
  --min-instances=0 \
  --max-instances=10 \
  --memory=1Gi \
  --cpu=1 \
  --timeout=300 \
  --port=8080 \
  --set-env-vars="DATABASE_URL=...,CELERY_BROKER_URL=...,GCS_BUCKET=...,PROJECT_ID=...,LOCATION=asia-northeast1" \
  --allow-unauthenticated
```

### Step 5: Deploy Dashboard

```bash
# Build
cd dashboard && npm run build

# Upload to Cloud Storage
gsutil -m rsync -R dist/ gs://webai-dashboard/

# Set up Cloud CDN via load balancer
gcloud compute backend-buckets create webai-dashboard-backend \
  --gcs-bucket-name=webai-dashboard \
  --enable-cdn
```

### Step 6: Deploy Celery Worker on GCE

```bash
gcloud compute instances create-with-container webai-worker \
  --zone=asia-northeast1-a \
  --machine-type=e2-micro \
  --container-image=asia-northeast1-docker.pkg.dev/PROJECT_ID/webai/worker \
  --container-env="DATABASE_URL=...,CELERY_BROKER_URL=...,GCS_BUCKET=...,PROJECT_ID=...,LOCATION=asia-northeast1,GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json"
```

### Step 7: Database Migration

```bash
# Run from local machine or Cloud Shell with Neon connection string
cd backend
DATABASE_URL="postgresql://..." alembic upgrade head
```

### Step 8: DNS + SSL

```bash
# API domain
gcloud run domain-mappings create \
  --service=webai-api \
  --domain=api.yourdomain.com \
  --region=asia-northeast1

# Dashboard domain → load balancer IP
# SSL auto-provisioned by Google-managed certificates
```

### Step 9: Update Webhook URLs

- LINE Developer Console: set webhook URL to `https://api.yourdomain.com/webhooks/line/{bot_id}`
- Meta Developer Console: set webhook URL to `https://api.yourdomain.com/webhooks/instagram/{bot_id}`
- Auth0: update callback URLs to production domain

---

## Post-Deploy Checklist

- [ ] `curl https://api.yourdomain.com/health` returns `{"status": "OK"}`
- [ ] Dashboard loads, Auth0 login works
- [ ] Create test bot, crawl a URL, verify Celery job completes
- [ ] Test web widget chat end-to-end
- [ ] Test LINE webhook (message → typing → response)
- [ ] Test Instagram webhook
- [ ] Verify GCS uploads work (PDF upload, crawled content storage)
- [ ] Set up Cloud Monitoring alerts (error rate > 5%, instance count near max)
- [ ] Verify Neon backups are enabled (automatic on paid plan)

---

## Scaling Playbook

| Trigger | Action | Cost Impact |
|---------|--------|-------------|
| Cold starts annoying users | Cloud Run `--min-instances=1` | +$5-10/month |
| Database connection limits | Upgrade Neon plan, enable connection pooling | +$10-20/month |
| Celery jobs backing up | Upgrade GCE to e2-small, or add second VM | +$7-15/month |
| >1000 concurrent chat users | Increase Cloud Run `--max-instances`, upgrade Neon | Variable |
| >5000 concurrent users | Replace Celery+Redis with Cloud Tasks + Cloud Run Jobs (eliminate always-on VM) | Saves VM cost |
| Global expansion beyond Japan | Add Cloud Run deployments in other regions | Per-region cost |

---

## Recent Bug Fixes (Already Applied)

### Event Loop Blocking Fix

All synchronous blocking calls to Vertex AI were wrapped in `asyncio.to_thread()` to prevent event loop blocking:

**Files changed:**
- `backend/api/routes/line_webhook.py` — `ensure_bot_corpus()` and `run_vertex_rag()` wrapped
- `backend/api/routes/saas.py` — All 5 call sites wrapped:
  - `/v1/pk/{key}/chat` endpoint (non-streaming)
  - `/v1/pk/{key}/chat/stream` endpoint (corpus lookup)
  - `/v1/org/bots/{bot_id}/test-chat` endpoint
  - `/v1/org/bots/{bot_id}/generate-system-prompt` endpoint
  - `/v1/org/bots/{bot_id}/generate-suggested-messages` endpoint

**Impact:** Before this fix, a single Gemini API call (~3-5 seconds) would block ALL concurrent requests — LINE, Web, everything. Now all requests process in parallel.

### Skip/Continue Button Logic Fix

Fixed bot creation flow where Skip and Continue buttons performed identical actions. Now:
- **Skip** clears/discards the current page's state and advances
- **Continue** persists the current selections and advances
- Applied across: CreateBotUrlsPage, CreateBotAdditionalSourcesPage, CreateBotEmbedPage

### Reservation Page UX Simplification

Simplified "Choose where reservation taps should go" page to human-friendly language with clear visual display of current platform link.
