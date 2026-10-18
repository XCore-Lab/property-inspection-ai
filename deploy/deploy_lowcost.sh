#!/usr/bin/env bash
# Low-cost Google Cloud deployment: a CPU-only Cloud Run SERVICE for the web UI/API (request-based billing, scales to
# zero, free tier) + a Cloud Run JOB with an NVIDIA L4 that processes one walkthrough per execution (the GPU is billed
# only while a job runs, no idle tail). Gemini through an AI Studio key in Secret Manager; no BigQuery / Pub/Sub /
# Vertex. deploy/deploy.sh is the all-in-one GPU-service variant from the hackathon.
#
#   PROJECT_ID=my-project DRIVE_ID=<Google_Hackathon_v1.zip Drive file id> ./deploy/deploy_lowcost.sh
#
# Before the first run, create the two secrets in the project:
#   printf '%s' "$GEMINI_KEY"   | gcloud secrets create gemini-api-key --replication-policy=automatic --data-file=-
#   printf '%s' "$UI_PASSWORD"  | gcloud secrets create app-api-key   --replication-policy=automatic --data-file=-
# First run: fetches the weights from Drive inside Cloud Build (~20 min), builds the base image (~40-60 min), the app
# image, the job and the service. Later runs: app image + redeploys only (a few minutes).
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"          # Cloud Run L4 regions: us-central1, us-east4, europe-west1, europe-west4, asia-southeast1
SERVICE="${SERVICE:-property-inspection-ai}"
JOB="${JOB:-property-inspection-gpu}"
BUCKET="${BUCKET:-${PROJECT_ID}-property-inspection}"     # app data (uploads, results, job status): /app/data on both
BUILD_BUCKET="${BUILD_BUCKET:-${BUCKET}-build}"            # model weights, only for the base image build
DRIVE_ID="${DRIVE_ID:-}"                                   # Google Drive id of Google_Hackathon_v1.zip (first run)
GEMINI_SECRET="${GEMINI_SECRET:-gemini-api-key}"
APP_API_KEY_SECRET="${APP_API_KEY_SECRET:-app-api-key}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.8-flash}"
DA3_MODEL="${DA3_MODEL:-DA3-BASE}"
JOB_CPU="${JOB_CPU:-8}"
JOB_MEM="${JOB_MEM:-32Gi}"                                 # L4 minimum is 4 / 16Gi (~25% cheaper per run; test first)
BUILD_BASE="${BUILD_BASE:-auto}"                           # auto = only if the base image is missing | 1 | 0
BASE_MACHINE="${BASE_MACHINE:-E2_HIGHCPU_32}"              # E2_HIGHCPU_8 if the project may not use 32 vCPU
DELETE_WEIGHTS="${DELETE_WEIGHTS:-1}"                      # drop the 9 GB staging bucket once the base image exists
REPO="${REPO:-property-inspection}"
SA_NAME="${SERVICE}-sa"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"
BASE_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:base"
export CLOUDSDK_CORE_PROJECT="$PROJECT_ID"                 # leaves the user's gcloud default project alone
cd "$(dirname "$0")/.."

echo "== APIs"
gcloud services enable run.googleapis.com storage.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com compute.googleapis.com

echo "== Artifact Registry, data bucket"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION"
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --location "$REGION" --uniform-bucket-level-access

echo "== service accounts"
PN=$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name "Property Inspection AI"
bind() {  # IAM on a fresh project is eventually consistent: one quiet retry
  "$@" >/dev/null 2>&1 || { sleep 15; "$@" >/dev/null; }
}
bind gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${SA}" --role roles/logging.logWriter --condition None
bind gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:${SA}" --role roles/storage.objectAdmin
for sec in "$GEMINI_SECRET" "$APP_API_KEY_SECRET"; do
  bind gcloud secrets add-iam-policy-binding "$sec" --member "serviceAccount:${SA}" --role roles/secretmanager.secretAccessor
done
CSA="${PN}-compute@developer.gserviceaccount.com"          # Cloud Build runs as this account on new projects
for role in roles/logging.logWriter roles/artifactregistry.writer roles/storage.objectAdmin; do
  bind gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${CSA}" --role "$role" --condition None
done

if [ "$BUILD_BASE" = "1" ] || { [ "$BUILD_BASE" = "auto" ] && \
     ! gcloud artifacts docker images describe "$BASE_IMAGE" >/dev/null 2>&1; }; then
  if ! gcloud storage ls "gs://${BUILD_BUCKET}/models/" >/dev/null 2>&1; then
    [ -n "$DRIVE_ID" ] || { echo "no staged weights in gs://${BUILD_BUCKET}/models and DRIVE_ID is not set"; exit 1; }
    gcloud storage buckets describe "gs://${BUILD_BUCKET}" >/dev/null 2>&1 || \
      gcloud storage buckets create "gs://${BUILD_BUCKET}" --location "$REGION" --uniform-bucket-level-access
    echo "== fetching the model weights from Google Drive inside Cloud Build (~20 min)"
    gcloud builds submit --no-source --config deploy/cloudbuild-fetch-models.yaml \
      --substitutions "_DRIVE_ID=${DRIVE_ID},_BUILD_BUCKET=${BUILD_BUCKET}"
  fi
  BASE_CFG=deploy/cloudbuild-base.yaml
  SUBS="_BASE_IMAGE=${BASE_IMAGE},_MODELS=gs://${BUILD_BUCKET}/models"
  if [ "$BASE_MACHINE" != "E2_HIGHCPU_32" ]; then
    BASE_CFG=deploy/.cloudbuild-base.tmp.yaml
    sed "s/E2_HIGHCPU_32/${BASE_MACHINE}/" deploy/cloudbuild-base.yaml > "$BASE_CFG"
    SUBS="${SUBS},_JOBS=4"
  fi
  echo "== building the BASE image on ${BASE_MACHINE} (CUDA compile + torch + weights; ~40-60 min)"
  gcloud builds submit --config "$BASE_CFG" --ignore-file deploy/base.gcloudignore --substitutions "$SUBS" .
  [ "$BASE_CFG" = deploy/cloudbuild-base.yaml ] || rm -f "$BASE_CFG"
fi
if [ "$DELETE_WEIGHTS" = "1" ] && gcloud storage buckets describe "gs://${BUILD_BUCKET}" >/dev/null 2>&1; then
  echo "== deleting the weights staging bucket (the base image holds them; a rebuild re-fetches from Drive)"
  gcloud storage rm -r "gs://${BUILD_BUCKET}"
fi

echo "== building the app image (code only, FROM ${BASE_IMAGE})"
gcloud builds submit --config deploy/cloudbuild.yaml --ignore-file deploy/app.gcloudignore \
  --substitutions "_IMAGE=${IMAGE},_BASE_IMAGE=${BASE_IMAGE}" .

VOL=(--add-volume "name=data,type=cloud-storage,bucket=${BUCKET}" --add-volume-mount "volume=data,mount-path=/app/data")
ENV_COMMON="PI_DATA_DIR=/app/data,PI_GEMINI_VERTEX=0,PI_GEMINI_MODEL=${GEMINI_MODEL},PI_DA3_MODEL=${DA3_MODEL}"

echo "== Cloud Run JOB: one walkthrough per execution on an NVIDIA L4"
gcloud run jobs deploy "$JOB" --image "$IMAGE" --region "$REGION" --service-account "$SA" \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu "$JOB_CPU" --memory "$JOB_MEM" \
  --tasks 1 --parallelism 1 --max-retries 0 --task-timeout 3600 \
  --command python --args=-m,backend.jobs.walkthrough_job \
  --set-env-vars "${ENV_COMMON},PI_GPU_RESIDENCY=coexist" \
  --set-secrets "PI_GEMINI_API_KEY=${GEMINI_SECRET}:latest" "${VOL[@]}"
bind gcloud run jobs add-iam-policy-binding "$JOB" --region "$REGION" --member "serviceAccount:${SA}" --role roles/run.developer

echo "== Cloud Run SERVICE: web UI + API, CPU only, scales to zero"
gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" --service-account "$SA" \
  --no-allow-unauthenticated --no-invoker-iam-check \
  --cpu 1 --memory 2Gi --cpu-boost --min-instances 0 --max-instances 2 --concurrency 40 --timeout 600 \
  --execution-environment gen2 --use-http2 \
  --set-env-vars "${ENV_COMMON},PI_GPU_JOB=projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB},PI_LOCATE_BACKEND=gemini,PI_VERIFY_BACKEND=gemini,PI_VALIDATE_BACKEND=off" \
  --set-secrets "PI_GEMINI_API_KEY=${GEMINI_SECRET}:latest,PI_API_KEY=${APP_API_KEY_SECRET}:latest" "${VOL[@]}"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo
echo "Deployed: ${URL}   (UI password: Secret Manager ${APP_API_KEY_SECRET})"
echo "Health:   ${URL}/api/v1/health"
echo "GPU jobs: gcloud run jobs executions list --job ${JOB} --region ${REGION}"
