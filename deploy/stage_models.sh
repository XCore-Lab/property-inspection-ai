#!/usr/bin/env bash
# One-time upload of the model weights (models/, ~9 GB) to a Cloud Storage bucket, for the base image build.
# Resumable: if the connection drops, run it again; files already uploaded are skipped. deploy/deploy.sh runs it too
# (then it only checks), so you can start this early, while you do the other setup steps.
#
#   PROJECT_ID=my-project ./deploy/stage_models.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT_ID}-property-inspection}"
BUILD_BUCKET="${BUILD_BUCKET:-${BUCKET}-build}"
cd "$(dirname "$0")/.."

[ -d models ] || { echo "models/ not found in $(pwd)"; exit 1; }
gcloud storage buckets describe "gs://${BUILD_BUCKET}" --project "$PROJECT_ID" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUILD_BUCKET}" --project "$PROJECT_ID" --location "$REGION" --uniform-bucket-level-access
echo "== uploading models/ to gs://${BUILD_BUCKET}/models (safe to re-run: finished files are skipped)"
gcloud storage rsync --recursive models "gs://${BUILD_BUCKET}/models"
echo "== model weights staged in gs://${BUILD_BUCKET}/models"
