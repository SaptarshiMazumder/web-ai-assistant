#!/bin/bash
# ==================================================================
# Web-AI Production Deployment Script
# ==================================================================
#
# USAGE:
#   1. Copy backend/.env.production.template → backend/.env.production
#   2. Fill in all values in .env.production
#   3. Fill in the CONFIG section below
#   4. Run:  bash deploy.sh          (full deploy)
#            bash deploy.sh build    (just build images)
#            bash deploy.sh api      (just deploy API)
#            bash deploy.sh worker   (just deploy worker)
#            bash deploy.sh migrate  (just run migrations)
#
# PREREQUISITES:
#   - gcloud CLI installed and authenticated (https://cloud.google.com/sdk)
#   - Git installed
#
# ==================================================================

set -euo pipefail

# ==================================================================
# CONFIG — Edit these values
# ==================================================================

PROJECT_ID="gen-lang-client-0545494042"                          # Your GCP project ID (e.g. gen-lang-client-0545494042)
REGION="asia-northeast1"               # Tokyo — don't change unless you have a reason
ZONE="${REGION}-a"

# Production environment file
ENV_FILE="backend/.env.production"

# Dashboard build args (baked into frontend JS at build time)
VITE_API_BASE="https://webai-api-wbifmyiivq-an.a.run.app"
VITE_AUTH0_DOMAIN="web-ai-agent.us.auth0.com"                    # From your .env.production
VITE_AUTH0_CLIENT_ID="tPGV5qoXCaS3IVn9rcbivnl8iMM4sJHD"         # From your .env.production
VITE_AUTH0_AUDIENCE="https://api.web-ai"                         # From your .env.production
VITE_AUTH_ORG_CLAIM="https://web-ai/org_id"                      # From your .env.production
VITE_SUPER_ADMIN_EMAILS="marcusrashford019.mufc@gmail.com"       # From your .env.production


# Cloud Run settings
API_SERVICE="webai-api"
API_MIN_INSTANCES=0                    # 0 = scale to zero (free when idle), 1 = always warm
API_MAX_INSTANCES=10
API_MEMORY="1Gi"
API_CPU=1
API_TIMEOUT=300

# Worker VM settings
WORKER_VM="webai-worker"
WORKER_MACHINE_TYPE="e2-small"         # ~$15/month in Tokyo
WORKER_DISK_SIZE="30GB"

# Custom domain (optional — leave empty to use Cloud Run default URL)
API_DOMAIN=""                          # e.g. api.yourdomain.com

# ==================================================================
# END CONFIG — Don't edit below unless you know what you're doing
# ==================================================================

# Derived values
REPO_NAME="webai"
SA_NAME="webai-sa"
API_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/api:latest"
WORKER_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/worker:latest"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[DEPLOY]${NC} $1"; }
ok()   { echo -e "${GREEN}  [OK]${NC} $1"; }
warn() { echo -e "${YELLOW}  [!]${NC} $1"; }
fail() { echo -e "${RED}  [FAIL]${NC} $1"; exit 1; }

# ==================================================================
# ENV FILE PARSER
# ==================================================================

# Parse a .env file line into KEY=VALUE (handles spaces, quotes, comments)
# Usage: parse_env_file FILE | while read line; do ... done
parse_env_file() {
  local file="$1"
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    # Skip comments and empty lines
    [[ -z "$line" || "$line" == \#* ]] && continue
    # Extract key (everything before first =, trimmed)
    local key="${line%%=*}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ -z "$key" ]] && continue
    # Extract value (everything after first =, trimmed)
    local value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    # Strip surrounding quotes
    if [[ "$value" =~ ^\"(.*)\"$ ]]; then
      value="${BASH_REMATCH[1]}"
    elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
      value="${BASH_REMATCH[1]}"
    fi
    echo "${key}=${value}"
  done < "$file"
}

# Load .env.production into shell variables
load_env() {
  [[ ! -f "$ENV_FILE" ]] && fail "$ENV_FILE not found. Copy from backend/.env.production.template"
  while IFS='=' read -r key value; do
    export "$key=$value" 2>/dev/null || true
  done < <(parse_env_file "$ENV_FILE")
  ok "Loaded $ENV_FILE"
}

# Generate Cloud Run --env-vars-file (YAML format)
generate_cloudrun_env_yaml() {
  local yaml_file="$1"
  > "$yaml_file"
  while IFS='=' read -r key value; do
    [[ "$key" == "GOOGLE_APPLICATION_CREDENTIALS" ]] && continue
    # Escape backslashes and double quotes for YAML
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    echo "${key}: \"${value}\"" >> "$yaml_file"
  done < <(parse_env_file "$ENV_FILE")
}

# Generate GCE --container-env flags as array
generate_gce_env_flags() {
  local flags=()
  while IFS='=' read -r key value; do
    [[ "$key" == "GOOGLE_APPLICATION_CREDENTIALS" ]] && continue
    flags+=("--container-env=${key}=${value}")
  done < <(parse_env_file "$ENV_FILE")
  # Return array elements, newline-separated
  printf '%s\n' "${flags[@]}"
}

# ==================================================================
# STEP 1: PREFLIGHT CHECKS
# ==================================================================
preflight() {
  log "Running preflight checks..."

  # Check gcloud CLI
  if ! command -v gcloud &>/dev/null; then
    fail "gcloud CLI not installed. Get it: https://cloud.google.com/sdk/docs/install"
  fi
  ok "gcloud CLI found: $(gcloud --version 2>/dev/null | head -1)"

  # Check config
  [[ -z "$PROJECT_ID" ]] && fail "PROJECT_ID is empty in deploy.sh. Set it to your GCP project ID."
  [[ -z "$REGION" ]]     && fail "REGION is empty in deploy.sh"

  # Check .env.production
  [[ ! -f "$ENV_FILE" ]] && fail "$ENV_FILE not found. Run: cp backend/.env.production.template backend/.env.production"

  # Verify critical env vars exist
  load_env
  [[ -z "${DATABASE_URL:-}" ]]      && fail "DATABASE_URL is empty in $ENV_FILE"
  [[ -z "${CELERY_BROKER_URL:-}" ]] && fail "CELERY_BROKER_URL is empty in $ENV_FILE"
  [[ -z "${PROJECT_ID:-}" ]]        && fail "PROJECT_ID is empty in $ENV_FILE"
  ok "Critical env vars present"

  # Check gcloud auth
  if ! gcloud auth print-access-token &>/dev/null 2>&1; then
    warn "Not authenticated. Opening browser for login..."
    gcloud auth login
  fi
  ok "gcloud authenticated"

  # Set project
  gcloud config set project "$PROJECT_ID" --quiet
  ok "GCP project: $PROJECT_ID"

  echo ""
  log "Preflight checks passed!"
  echo ""
}

# ==================================================================
# STEP 2: GCP SETUP (APIs, Artifact Registry, Service Account)
# ==================================================================
setup_gcp() {
  log "Setting up GCP infrastructure..."

  # Enable required APIs
  log "Enabling APIs (may take 1-2 minutes on first run)..."
  gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    compute.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    --quiet
  ok "APIs enabled"

  # Create Artifact Registry
  if ! gcloud artifacts repositories describe "$REPO_NAME" \
    --location="$REGION" &>/dev/null 2>&1; then
    log "Creating Artifact Registry..."
    gcloud artifacts repositories create "$REPO_NAME" \
      --repository-format=docker \
      --location="$REGION" \
      --description="Web-AI Docker images" \
      --quiet
    ok "Artifact Registry created"
  else
    ok "Artifact Registry already exists"
  fi

  # Create service account
  if ! gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null 2>&1; then
    log "Creating service account..."
    gcloud iam service-accounts create "$SA_NAME" \
      --display-name="Web-AI Service Account" \
      --quiet
    ok "Service account created: $SA_EMAIL"
  else
    ok "Service account already exists"
  fi

  # Grant roles (idempotent)
  local roles=(
    "roles/aiplatform.user"
    "roles/storage.objectAdmin"
    "roles/artifactregistry.reader"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
  )
  for role in "${roles[@]}"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
      --member="serviceAccount:${SA_EMAIL}" \
      --role="$role" \
      --condition=None \
      --quiet >/dev/null 2>&1
  done
  ok "Service account roles granted"

  # Configure Docker auth
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet 2>/dev/null
  ok "Docker auth configured for Artifact Registry"

  echo ""
  log "GCP setup complete!"
  echo ""
}

# ==================================================================
# STEP 3: BUILD IMAGES (via Cloud Build)
# ==================================================================
build() {
  log "Building Docker images via Cloud Build..."
  log "Both API and Worker images build in parallel (~5-10 min)..."
  echo ""

  gcloud builds submit \
    --project="$PROJECT_ID" \
    --config=cloudbuild.yaml \
    --substitutions="\
_REGION=${REGION},\
_REPO=${REPO_NAME},\
_TAG=latest,\
_VITE_API_BASE=${VITE_API_BASE},\
_VITE_AUTH0_DOMAIN=${VITE_AUTH0_DOMAIN},\
_VITE_AUTH0_CLIENT_ID=${VITE_AUTH0_CLIENT_ID},\
_VITE_AUTH0_AUDIENCE=${VITE_AUTH0_AUDIENCE},\
_VITE_AUTH_ORG_CLAIM=${VITE_AUTH_ORG_CLAIM},\
_VITE_SUPER_ADMIN_EMAILS=${VITE_SUPER_ADMIN_EMAILS}"

  echo ""
  ok "API image:    $API_IMAGE"
  ok "Worker image: $WORKER_IMAGE"
  echo ""
}

# ==================================================================
# STEP 4: DEPLOY API TO CLOUD RUN
# ==================================================================
api() {
  log "Deploying API to Cloud Run..."

  # Generate env vars YAML
  local yaml_file="/tmp/webai-cloudrun-env.yaml"
  generate_cloudrun_env_yaml "$yaml_file"
  ok "Generated env vars file ($yaml_file)"

  gcloud run deploy "$API_SERVICE" \
    --image="$API_IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --service-account="$SA_EMAIL" \
    --min-instances="$API_MIN_INSTANCES" \
    --max-instances="$API_MAX_INSTANCES" \
    --memory="$API_MEMORY" \
    --cpu="$API_CPU" \
    --timeout="$API_TIMEOUT" \
    --port=8080 \
    --env-vars-file="$yaml_file" \
    --allow-unauthenticated \
    --quiet

  # Get the deployed URL
  local api_url
  api_url=$(gcloud run services describe "$API_SERVICE" \
    --region="$REGION" \
    --format='value(status.url)')

  echo ""
  ok "API deployed: $api_url"

  # Custom domain mapping
  if [[ -n "$API_DOMAIN" ]]; then
    log "Mapping custom domain: $API_DOMAIN"
    gcloud run domain-mappings create \
      --service="$API_SERVICE" \
      --domain="$API_DOMAIN" \
      --region="$REGION" \
      --quiet 2>/dev/null || ok "Domain mapping already exists"
    ok "Domain mapped. Add these DNS records:"
    gcloud run domain-mappings describe \
      --domain="$API_DOMAIN" \
      --region="$REGION" \
      --format='table(resourceRecords.type, resourceRecords.rrdata)'
  fi

  # Warn if VITE_API_BASE wasn't set
  if [[ -z "$VITE_API_BASE" ]]; then
    echo ""
    warn "VITE_API_BASE was empty during build."
    warn "Dashboard will use relative URLs (works if served from same Cloud Run service)."
    warn "If you need an explicit URL, set VITE_API_BASE=$api_url and run: bash deploy.sh build api"
  fi

  # Clean up
  rm -f "$yaml_file"
  echo ""
}

# ==================================================================
# STEP 5: DEPLOY WORKER TO GCE VM
# ==================================================================
worker() {
  log "Deploying Celery worker to GCE VM..."

  # Read env flags into array
  local env_flags=()
  while IFS= read -r flag; do
    env_flags+=("$flag")
  done < <(generate_gce_env_flags)

  # Check if VM exists
  if gcloud compute instances describe "$WORKER_VM" \
    --zone="$ZONE" &>/dev/null 2>&1; then
    log "Updating existing worker VM..."
    gcloud compute instances update-container "$WORKER_VM" \
      --zone="$ZONE" \
      --container-image="$WORKER_IMAGE" \
      "${env_flags[@]}" \
      --quiet
    ok "Worker VM updated"
  else
    log "Creating worker VM (${WORKER_MACHINE_TYPE}, ${WORKER_DISK_SIZE})..."
    gcloud compute instances create-with-container "$WORKER_VM" \
      --zone="$ZONE" \
      --machine-type="$WORKER_MACHINE_TYPE" \
      --boot-disk-size="$WORKER_DISK_SIZE" \
      --container-image="$WORKER_IMAGE" \
      --service-account="$SA_EMAIL" \
      --scopes=cloud-platform \
      --tags=webai-worker \
      "${env_flags[@]}" \
      --quiet
    ok "Worker VM created"
  fi

  echo ""
  ok "Worker VM: $WORKER_VM ($ZONE)"
  echo ""
}

# ==================================================================
# STEP 6: RUN DATABASE MIGRATIONS
# ==================================================================
migrate() {
  log "Running database migrations via Cloud Run job..."

  load_env

  # Create or update migration job
  if gcloud run jobs describe webai-migrate \
    --region="$REGION" &>/dev/null 2>&1; then
    gcloud run jobs update webai-migrate \
      --image="$API_IMAGE" \
      --region="$REGION" \
      --service-account="$SA_EMAIL" \
      --set-env-vars="DATABASE_URL=${DATABASE_URL}" \
      --command="alembic" \
      --args="upgrade,head" \
      --max-retries=0 \
      --quiet
  else
    gcloud run jobs create webai-migrate \
      --image="$API_IMAGE" \
      --region="$REGION" \
      --service-account="$SA_EMAIL" \
      --set-env-vars="DATABASE_URL=${DATABASE_URL}" \
      --command="alembic" \
      --args="upgrade,head" \
      --max-retries=0 \
      --quiet
  fi

  # Execute migration
  gcloud run jobs execute webai-migrate \
    --region="$REGION" \
    --wait \
    --quiet

  ok "Database migrations complete"
  echo ""
}

# ==================================================================
# STEP 7: VERIFY
# ==================================================================
verify() {
  log "Verifying deployment..."

  # Get API URL
  local api_url
  api_url=$(gcloud run services describe "$API_SERVICE" \
    --region="$REGION" \
    --format='value(status.url)' 2>/dev/null || echo "")

  if [[ -z "$api_url" ]]; then
    warn "Could not get API URL. Is the service deployed?"
    return
  fi

  # Health check
  log "Checking API health..."
  local health_status
  health_status=$(curl -s -o /dev/null -w '%{http_code}' "${api_url}/health" --max-time 30 || echo "000")

  if [[ "$health_status" == "200" ]]; then
    ok "API health check passed (${api_url}/health → 200)"
  else
    warn "API health check returned $health_status (may be cold-starting, try again in 10s)"
  fi

  # Liveness check
  local live_status
  live_status=$(curl -s -o /dev/null -w '%{http_code}' "${api_url}/live" --max-time 10 || echo "000")

  if [[ "$live_status" == "200" ]]; then
    ok "API liveness check passed (${api_url}/live → 200)"
  else
    warn "API liveness check returned $live_status"
  fi

  # Worker VM check
  local worker_status
  worker_status=$(gcloud compute instances describe "$WORKER_VM" \
    --zone="$ZONE" \
    --format='value(status)' 2>/dev/null || echo "NOT_FOUND")

  if [[ "$worker_status" == "RUNNING" ]]; then
    ok "Worker VM is RUNNING"
  else
    warn "Worker VM status: $worker_status"
  fi

  echo ""
}

# ==================================================================
# STEP 8: PRINT SUMMARY
# ==================================================================
summary() {
  local api_url
  api_url=$(gcloud run services describe "$API_SERVICE" \
    --region="$REGION" \
    --format='value(status.url)' 2>/dev/null || echo "(not deployed yet)")

  echo ""
  echo "=================================================================="
  echo -e "${GREEN}  DEPLOYMENT COMPLETE${NC}"
  echo "=================================================================="
  echo ""
  echo "  API:        ${api_url}"
  echo "  Dashboard:  ${api_url}/dashboard/"
  echo "  Health:     ${api_url}/health"
  echo "  Worker VM:  ${WORKER_VM} (${ZONE})"
  echo ""
  echo "  Images:"
  echo "    API:      ${API_IMAGE}"
  echo "    Worker:   ${WORKER_IMAGE}"
  echo ""
  echo "  Scaling:"
  echo "    API:      ${API_MIN_INSTANCES}→${API_MAX_INSTANCES} instances (Cloud Run)"
  echo "    Worker:   1 instance (${WORKER_MACHINE_TYPE})"
  echo ""
  if [[ -n "$API_DOMAIN" ]]; then
    echo "  Custom domain: https://${API_DOMAIN}"
    echo "    (Make sure DNS records are configured)"
    echo ""
  fi
  echo "  Next steps:"
  echo "    - Update webhook URLs in LINE/Instagram developer consoles"
  echo "    - Update Auth0 callback URLs to production domain"
  echo "    - Test: create a bot, crawl a URL, chat with it"
  echo ""
  echo "  Useful commands:"
  echo "    View API logs:     gcloud run services logs read ${API_SERVICE} --region=${REGION}"
  echo "    View worker logs:  gcloud compute ssh ${WORKER_VM} --zone=${ZONE} --command='sudo docker logs webai-worker'"
  echo "    Redeploy API:      bash deploy.sh build api"
  echo "    Redeploy worker:   bash deploy.sh build worker"
  echo "    Scale API:         gcloud run services update ${API_SERVICE} --region=${REGION} --min-instances=1"
  echo "    SSH to worker:     gcloud compute ssh ${WORKER_VM} --zone=${ZONE}"
  echo ""
  echo "=================================================================="
}

# ==================================================================
# MAIN
# ==================================================================
full_deploy() {
  preflight
  setup_gcp
  build
  api
  worker
  migrate
  verify
  summary
}

# Allow running individual steps: bash deploy.sh build, bash deploy.sh api, etc.
if [[ $# -eq 0 ]]; then
  full_deploy
else
  # Run each argument as a function
  for step in "$@"; do
    if declare -f "$step" >/dev/null 2>&1; then
      "$step"
    else
      fail "Unknown step: $step. Available: preflight, setup_gcp, build, api, worker, migrate, verify, summary"
    fi
  done
fi
