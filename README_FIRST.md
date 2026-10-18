# Property Inspection AI – Google Hackathon v1

Everything needed is in this folder: code, web UI, 3D engine, all model weights and the GPU programs.
**No API key is included.** In the cloud the app uses Vertex AI through its service account, so it needs no key.

> **Cloned from GitHub?** The repository holds the code only. The model weights (`models/`, ~9 GB) and the
> Windows GPU programs (`bin/`, ~2 GB) are too large for Git: copy both folders from `Google_Hackathon_v1.zip`
> into this folder before running or deploying. Local settings go in `.env` (copy `.env.example`); it is never committed.

## What's inside

| Folder | What | Size |
|---|---|---|
| `backend/`, `frontend/`, `engine3d/`, `depth_anything_3/`, `scripts/` | the app (FastAPI + web UI + 3D engine) | small |
| `models/locate_anything/` | LocateAnything-3B (finds defect boxes) | 3.9 GB |
| `models/gemma4/` | Gemma 4 E4B + vision projector (checks each box) | 5.2 GB |
| `models/hf/` | Depth Anything 3 BASE (video → 3D) | 0.5 GB |
| `third_party/locate-anything.cpp/` | source code; the cloud build compiles it for Linux + CUDA | small |
| `bin/` | Windows GPU programs, **local Windows runs only** (never uploaded to the cloud) | 2 GB |
| `Dockerfile`, `deploy/` | cloud container + one-command deploy script | |

AI roles: **Gemini 3.8 Flash** (Vertex AI) identifies the room and its parts and writes the summary and Q&A.
**LocateAnything-3B** finds possible defects, and **Gemma 4 E4B** confirms each one (both on the GPU).
A rule engine then decides Detected / Needs review and works out the risk grade and repair cost.
**Depth Anything 3** + OpenCV SLAM build the 3D model from a video (Lite = faster build, Advanced = full detail).

## A. Deploy to Google Cloud (Cloud Run + NVIDIA L4 GPU)

Requirements:
1. A Google Cloud project with **billing** enabled.
2. **Cloud Run GPU quota** in the region: IAM & Admin → Quotas → "Total Nvidia L4 GPU allocation without
   zonal redundancy, per project per region" ≥ 1 in `us-central1`. Request it first; approval can take a while.
3. Access to the Gemini model on Vertex AI in that project (default `gemini-3.8-flash`; change it with
   `GEMINI_MODEL=...`).
4. A PC with the **Google Cloud CLI** installed (`gcloud` and `bq`), plus a bash shell
   (Linux / macOS, or **Git Bash** / **WSL** on Windows). Don't use Cloud Shell: its 5 GB disk is too small.

Steps:
```bash
gcloud auth login
cd Google_Hackathon_v1
PROJECT_ID=<your-project-id> ./deploy/deploy.sh
```

The script:
1. enables the APIs;
2. creates the storage bucket, BigQuery dataset, Pub/Sub topic and service account;
3. first deploy only: uploads this folder (~9 GB; `bin/` is skipped) to Cloud Build and builds the base image
   (compiles the GPU programs for Linux, installs PyTorch, bakes in all the models: **~40–60 min**);
4. builds the app image on top of it (uploads only the code: a few MB, a few minutes). Later deploys repeat only
   this step; `BUILD_BASE=1` rebuilds the base too (new model weights or GPU programs);
5. deploys Cloud Run with an L4 GPU and prints the **URL**.

Open the URL, then use **Video walkthroughs**: upload a phone video of the property and pick Lite or Advanced.

Options:
- `APP_API_KEY=<secret>`: the web page asks for this password once, so the public URL isn't open to everyone.
- `MIN_INSTANCES=1`: keeps the GPU warm for a live demo (no ~2–3 min cold start), but it's **billed while
  idle**. Set it back to 0 afterwards.
- `REGION=europe-west4` (or another region with L4 GPUs).
- `GPU=0`: CPU-only deploy. Not useful for this video-only version: video walkthroughs need the GPU, and
  single-photo uploads are off (`PI_PHOTO_UPLOAD=1` turns them back on).

Cost guide: the GPU service (8 vCPU, 32 GiB, one L4) costs about US$1.42 per hour while running and scales to
zero when idle with `MIN_INSTANCES=0`. A Lite walkthrough costs about US$1 including Gemini (about US$0.005 per
call). `MIN_INSTANCES=1` costs about US$34 per day, so set it back to 0 after a demo.

Check it's healthy: `<URL>/api/v1/health` should show `"cuda": true` and the localizer and VLM as available.
Logs: Cloud Run → service `property-inspection-ai` → Logs.

## B. Run locally on Windows (NVIDIA GPU, 6 GB+)

```bat
setup_local.bat          :: once: creates .venv (Python 3.11 required) and .env
:: edit .env -> PI_GEMINI_API_KEY=<AI Studio key>
run_local.bat            :: http://127.0.0.1:8080
```

## Notes
- Video walkthroughs take ~15–35 min each. Keep the page open while one runs.
- Licences: LocateAnything-3B weights use the **NVIDIA non-commercial licence**, and Gemma uses the Gemma terms.
  This is fine for a hackathon demo, but not for commercial use without other licensing. See `README.md` → Licensing.
- Full technical documentation: `README.md`.
