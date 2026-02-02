# Run and test

## Prerequisites

- **Docker** and **Docker Compose**
- **Backend env**: `backend/.env` with at least:
  - `OPENAI_API_KEY` (required)
  - `GOOGLE_API_KEY` (or GCP service account)
  - `DATABASE_URL` (Postgres, e.g. `postgresql://webai:Madrid2018@localhost:5432/webai` when using docker-compose Postgres)
  - `GOOGLE_APPLICATION_CREDENTIALS` path to a GCP service account JSON (for RAG/indexing)
- **GCP creds file**: path used in docker-compose as `HOST_GCP_CREDS` (default `./backend/creds/gen-lang-client-*.json`). Create the file and set the path if needed.

## Run with Docker (recommended)

From the project root:

```bash
# Build and start all services (backend, celery-worker, redis, postgres, dashboard)
docker-compose up --build
```

- **Backend API**: http://localhost:5000  
- **Dashboard**: http://localhost:5173 (served by the dashboard container on port 80, mapped to 5173)  
- **Postgres**: localhost:5432 (user `webai`, db `webai`, password `Madrid2018`)  
- **Redis**: localhost:6379  

Backend and celery-worker images include **URLFinder**; discovery will use it when you run URL discovery.

To run in the background:

```bash
docker-compose up --build -d
```

Logs:

```bash
docker-compose logs -f backend
docker-compose logs -f celery-worker
```

## Run locally (without Docker)

1. **Postgres + Redis** running (e.g. start them via Docker or install locally).
2. **Backend**  
   Create a venv, install deps, set env (e.g. from `backend/.env`), then:

   ```bash
   cd backend
   pip install -r requirements.txt
   python -m playwright install --with-deps chromium
   # Optional: install URLFinder for discovery (or skip; discovery will fall back to HTTP + crawl4ai)
   # go install -v github.com/projectdiscovery/urlfinder/cmd/urlfinder@latest
   export DATABASE_URL=postgresql://webai:Madrid2018@localhost:5432/webai
   uvicorn main:app --reload --host 0.0.0.0 --port 5000
   ```

3. **Dashboard** (separate terminal):

   ```bash
   cd dashboard
   npm install
   npm run dev
   ```

   Set `VITE_API_BASE=http://localhost:5000` if the API is on 5000. Open the URL Vite prints (e.g. http://localhost:5173).

## Test URL discovery

### From the dashboard

1. **Create Bot flow**  
   - Open the dashboard → Create Bot.  
   - Enter a **website URL** (e.g. `https://example.com`).  
   - Choose **Automatic** (uses URLFinder first, then HTTP + crawl4ai fallback) or **Sitemap**.  
   - Click **Discover URLs**.  
   - You should see URLs stream in; when done, select pages and continue.

2. **Knowledge tab**  
   - Open a bot → **Knowledge** tab.  
   - Enter a URL, choose **Automatic** or **Sitemap**, click **Discover pages**.  
   - Same streaming discovery; you can then add selected URLs to training.

If URLFinder is used, backend logs may show: `URLFinder found N URLs for ...`.

### With curl (stream endpoint)

Discovery is behind auth. Easiest is to use the dashboard. If you have a valid JWT:

```bash
curl -X POST "http://localhost:5000/v1/org/url-discovery/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT" \
  -d '{"url": "https://example.com", "method": "auto"}' \
  --no-buffer
```

You’ll see server-sent events: `discovered` (per URL) and `done` (with `method_used`, e.g. `urlfinder` or `sitemap`).

### Quick sanity check (no auth)

Health check:

```bash
curl -s http://localhost:5000/health
```

If the backend uses a public health route, you should get a 200. URL discovery itself requires auth (dashboard or JWT).

## Troubleshooting

- **“URLFinder binary not found”**  
  In Docker, the image installs URLFinder; no extra step. Locally, install with `go install .../urlfinder@latest` and ensure `urlfinder` is on PATH, or leave it unset — discovery will fall back to HTTP + crawl4ai.

- **Discovery returns 0 URLs**  
  For **Automatic**: URLFinder runs first (passive sources); if it finds nothing, HTTP crawl runs, then crawl4ai for JS-heavy sites. Try **Sitemap** if the site has a sitemap. Check backend/celery-worker logs for errors.

- **Database connection errors**  
  Ensure `DATABASE_URL` in `backend/.env` matches your Postgres (with Docker: `postgresql://webai:Madrid2018@postgres:5432/webai` inside the backend container; docker-compose sets hostname `postgres`). For local runs, use `localhost` in the URL.

- **GCP / RAG errors**  
  Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid JSON key and the path is correct in the container (volume mount in docker-compose).
