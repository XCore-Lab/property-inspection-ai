#!/usr/bin/env bash
# Copy finished walkthroughs (data/walkthrough/<name>/) into the Cloud Storage bucket that the service mounts at
# /app/data, so the deployed app lists and opens them without processing a video. This is how a CPU-only deploy
# (GPU=0, e.g. while the L4 GPU isn't available) still shows real results. Each folder is self-contained
# (3D models, coverage map, keyframes, report); ~30 MB for room_walkthrough_lite. Safe to re-run (only changes upload).
#
#   PROJECT_ID=my-project ./deploy/upload_walkthroughs.sh                          # every finished walkthrough
#   PROJECT_ID=my-project ./deploy/upload_walkthroughs.sh room_walkthrough_lite    # just these
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT_ID}-property-inspection}"      # same default as deploy/deploy.sh
cd "$(dirname "$0")/.."

names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  for f in data/walkthrough/*/walkthrough.json; do
    [ -f "$f" ] && names+=("$(basename "$(dirname "$f")")")
  done
fi
[ ${#names[@]} -gt 0 ] || { echo "no finished walkthroughs in data/walkthrough/"; exit 1; }

gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT_ID" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --project "$PROJECT_ID" --location "$REGION" --uniform-bucket-level-access
for n in "${names[@]}"; do
  src="data/walkthrough/${n}"
  [ -f "${src}/walkthrough.json" ] || { echo "skip ${n}: ${src}/walkthrough.json not found"; continue; }
  echo "== ${n} -> gs://${BUCKET}/walkthrough/${n}/"
  gcloud storage rsync --recursive "$src" "gs://${BUCKET}/walkthrough/${n}"
done
echo
echo "Done. The app lists them under Video walkthroughs (reload the page; no redeploy needed)."
