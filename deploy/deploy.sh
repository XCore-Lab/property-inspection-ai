#!/usr/bin/env bash
# One-shot Google Cloud deployment (run in Cloud Shell or any shell with gcloud + bq logged in).
#
#   PROJECT_ID=my-project ./deploy/deploy.sh            # GPU service (video walkthroughs)
#   PROJECT_ID=my-project GPU=0 ./deploy/deploy.sh      # CPU-only service: no video walkthroughs (UI testing only)
#   PROJECT_ID=my-project BUILD_BASE=1 ./deploy/deploy.sh   # also rebuild the base image (new weights / GPU binaries / torch)
#   PROJECT_ID=my-project BASE_MACHINE=E2_HIGHCPU_8 ./deploy/deploy.sh   # if Cloud Build refuses the 32-vCPU machine
#
# Two images: a BASE image (Dockerfile.base: CUDA builds, torch, ~9 GB of model weights) built on the first deploy
# or with BUILD_BASE=1, and the thin APP image (Dockerfile: code only) built on every deploy, which uploads only the
# code (deploy/app.gcloudignore skips models/) and pushes only the code layers. The weights reach the base build
# through a bucket (deploy/stage_models.sh, a resumable one-time upload), not through the build upload.
#
# Creates: Artifact Registry repo, Cloud Storage bucket (app data, mounted into Cloud Run), BigQuery dataset,
# Pub/Sub topic, a service account with least-privilege roles, builds the image with Cloud Build and deploys
# Cloud Run. Gemini runs on Vertex AI with the service account - no API key is stored anywhere.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"            # Cloud Run GPU regions: us-central1, europe-west1, europe-west4, asia-southeast1 ...
IMAGE_REGION="${IMAGE_REGION:-$REGION}"   # where images are built and stored: keep it (e.g. us-central1) when moving
                                          # REGION, so the service can move without rebuilding the base image
SERVICE="${SERVICE:-property-inspection-ai}"
GPU="${GPU:-1}"
BUCKET="${BUCKET:-${PROJECT_ID}-property-inspection}"
BQ_DATASET="${BQ_DATASET:-property_risk}"
TOPIC="${TOPIC:-inspection-events}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.8-flash}"
DA3_MODEL="${DA3_MODEL:-DA3-BASE}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"       # 1 = keep one GPU instance warm: no cold start, but ~$34/day (~$1.42/h) until set back to 0
APP_API_KEY="${APP_API_KEY:-}"             # optional: protect the public URL (UI asks for it once)
SA_NAME="${SERVICE}-sa"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO="${REPO:-property-inspection}"
IMAGE="${IMAGE_REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"
BASE_IMAGE="${IMAGE_REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:base"
BUILD_BASE="${BUILD_BASE:-auto}"          # auto = only if the base image doesn't exist yet | 1 = rebuild it | 0 = never
BUILD_BUCKET="${BUILD_BUCKET:-${BUCKET}-build}"   # model weights staged for the base build (deploy/stage_models.sh)
BASE_MACHINE="${BASE_MACHINE:-E2_HIGHCPU_32}"     # Cloud Build machine for the base build; E2_HIGHCPU_8 if 32 vCPU is refused
BASE_JOBS="${BASE_JOBS:-}"                        # parallel CUDA compile jobs ("" = all CPUs; 4 on E2_HIGHCPU_8)
cd "$(dirname "$0")/.."

gcloud config set project "$PROJECT_ID"
echo "== enabling APIs"
gcloud services enable run.googleapis.com aiplatform.googleapis.com storage.googleapis.com \
  bigquery.googleapis.com pubsub.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

echo "== Artifact Registry, bucket, BigQuery, Pub/Sub"
gcloud artifacts repositories describe "$REPO" --location "$IMAGE_REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$IMAGE_REGION"
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --location "$REGION" --uniform-bucket-level-access
bq --location=US show "${PROJECT_ID}:${BQ_DATASET}" >/dev/null 2>&1 || bq --location=US mk -d "${PROJECT_ID}:${BQ_DATASET}"
gcloud pubsub topics describe "$TOPIC" >/dev/null 2>&1 || gcloud pubsub topics create "$TOPIC"

echo "== service account"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name "Property Inspection AI"
for role in roles/aiplatform.user roles/storage.objectAdmin roles/bigquery.dataEditor roles/bigquery.jobUser \
            roles/pubsub.publisher roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${SA}" --role "$role" \
    --condition None >/dev/null
done

if [ "$BUILD_BASE" = "1" ] || { [ "$BUILD_BASE" = "auto" ] && \
     ! gcloud artifacts docker images describe "$BASE_IMAGE" >/dev/null 2>&1; }; then
  # model weights: one-time ~9 GB upload to a bucket (resumable; a re-run skips finished files)
  PROJECT_ID="$PROJECT_ID" REGION="$IMAGE_REGION" BUCKET="$BUCKET" BUILD_BUCKET="$BUILD_BUCKET" bash deploy/stage_models.sh
  BASE_CFG=deploy/cloudbuild-base.yaml
  SUBS="_BASE_IMAGE=${BASE_IMAGE},_MODELS=gs://${BUILD_BUCKET}/models"
  if [ "$BASE_MACHINE" != "E2_HIGHCPU_32" ]; then    # e.g. free trial (8 vCPU cap): smaller machine, fewer parallel compiles
    BASE_CFG=deploy/.cloudbuild-base.tmp.yaml
    sed "s/E2_HIGHCPU_32/${BASE_MACHINE}/" deploy/cloudbuild-base.yaml > "$BASE_CFG"
    SUBS="${SUBS},_JOBS=${BASE_JOBS:-4}"
  elif [ -n "$BASE_JOBS" ]; then
    SUBS="${SUBS},_JOBS=${BASE_JOBS}"
  fi
  echo "== building the BASE image with Cloud Build on ${BASE_MACHINE} (CUDA compile + PyTorch + weights; ~40-60 min)"
  gcloud builds submit --config "$BASE_CFG" --ignore-file deploy/base.gcloudignore --substitutions "$SUBS" .
  [ "$BASE_CFG" = deploy/cloudbuild-base.yaml ] || rm -f "$BASE_CFG"
fi
echo "== building the app image with Cloud Build (uploads only the code; FROM ${BASE_IMAGE})"
gcloud builds submit --config deploy/cloudbuild.yaml --ignore-file deploy/app.gcloudignore \
  --substitutions "_IMAGE=${IMAGE},_BASE_IMAGE=${BASE_IMAGE}" .

ENV="PI_GEMINI_VERTEX=1,PI_GCP_PROJECT=${PROJECT_ID},PI_GCP_LOCATION=global,PI_GEMINI_MODEL=${GEMINI_MODEL}"
ENV="${ENV},PI_DATA_DIR=/app/data,PI_BQ_DATASET=${BQ_DATASET},PI_PUBSUB_TOPIC=${TOPIC},PI_DA3_MODEL=${DA3_MODEL}"
if [ "$GPU" = "1" ]; then
  ENV="${ENV},PI_GPU_RESIDENCY=coexist"    # L4 has 24 GB: LocateAnything + Gemma stay loaded together
else
  ENV="${ENV},PI_LOCATE_BACKEND=gemini,PI_VERIFY_BACKEND=gemini,PI_VALIDATE_BACKEND=off"   # no GPU: all-Gemini photos
fi
[ -n "$APP_API_KEY" ] && ENV="${ENV},PI_API_KEY=${APP_API_KEY}"

COMMON=(--image "$IMAGE" --region "$REGION" --service-account "$SA" --allow-unauthenticated
        --set-env-vars "$ENV" --timeout 3600 --concurrency 20 --min-instances "$MIN_INSTANCES" --max-instances 1 --no-cpu-throttling
        --execution-environment gen2 --use-http2      # h2c to hypercorn: no 32 MiB HTTP/1 request cap on uploads
        --add-volume "name=data,type=cloud-storage,bucket=${BUCKET}" --add-volume-mount "volume=data,mount-path=/app/data")
if [ "$GPU" = "1" ]; then
  echo "== deploying Cloud Run with an NVIDIA L4 GPU"
  gcloud run deploy "$SERVICE" "${COMMON[@]}" --cpu 8 --memory 32Gi --gpu 1 --gpu-type nvidia-l4 \
    --no-gpu-zonal-redundancy
else
  echo "== deploying Cloud Run (CPU only: video walkthroughs need GPU=1; photo uploads are off unless PI_PHOTO_UPLOAD=1)"
  gcloud run deploy "$SERVICE" "${COMMON[@]}" --cpu 2 --memory 4Gi
fi

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo
echo "Deployed: ${URL}"
echo "Health:   ${URL}/api/v1/health"
echo "Data:     gs://${BUCKET}  |  BigQuery: ${PROJECT_ID}.${BQ_DATASET}.inspections  |  Pub/Sub: ${TOPIC}"
