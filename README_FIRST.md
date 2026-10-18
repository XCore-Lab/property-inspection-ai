# Property Inspection AI – Google Hackathon v1

Everything needed is in this folder: code, web UI, 3D engine, all model weights and the GPU programs.
**No API key is included.** Put your own Gemini key in `.env`.

> **Cloned from GitHub?** The repository holds the code only. The model weights (`models/`, ~9 GB) and the
> Windows GPU programs (`bin/`, ~2 GB) are too large for Git: copy both folders from `Google_Hackathon_v1.zip`
> into this folder before running. Local settings go in `.env` (copy `.env.example`); it is never committed.

## What's inside

| Folder | What | Size |
|---|---|---|
| `backend/`, `frontend/`, `engine3d/`, `depth_anything_3/`, `scripts/` | the app (FastAPI + web UI + 3D engine) | small |
| `models/locate_anything/` | LocateAnything-3B (finds defect boxes) | 3.9 GB |
| `models/gemma4/` | Gemma 4 E4B + vision projector (checks each box) | 5.2 GB |
| `models/hf/` | Depth Anything 3 BASE (video → 3D) | 0.5 GB |
| `third_party/locate-anything.cpp/` | source code of the LocateAnything runtime | small |
| `bin/` | Windows GPU programs for local runs | 2 GB |

AI roles: **Gemini 3.8 Flash** identifies the room and its parts and writes the summary and Q&A.
**LocateAnything-3B** finds possible defects, and **Gemma 4 E4B** confirms each one (both on the GPU).
A rule engine then decides Detected / Needs review and works out the risk grade and repair cost.
**Depth Anything 3** + OpenCV SLAM build the 3D model from a video (Lite = faster build, Advanced = full detail).

## Run locally on Windows (NVIDIA GPU, 6 GB+)

```bat
setup_local.bat          :: once: creates .venv (Python 3.11 required) and .env
:: edit .env -> PI_GEMINI_API_KEY=<AI Studio key>
run_local.bat            :: http://127.0.0.1:8080
```

## Deploy to Google Cloud (low cost)

One command deploys a CPU-only Cloud Run web service plus a Cloud Run Job with an NVIDIA L4 that runs only while a
video is processing (idle ≈ US$1.20/month, ≈ US$0.85 per walkthrough). The model weights are fetched from Google
Drive inside Cloud Build, so nothing large passes through your PC:

```bash
printf '%s' "$GEMINI_KEY" | gcloud secrets create gemini-api-key --replication-policy=automatic --data-file=-
JOB_REGION=us-east4 PROJECT_ID=<project-id> DRIVE_ID=<Google_Hackathon_v1.zip file id> ./deploy/deploy_lowcost.sh
```

Details, options and the architecture: `README.md` → *Deploy to Google Cloud (low cost)*.

## Notes
- Video walkthroughs take ~15–35 min each. Keep the page open while one runs.
- Licences: LocateAnything-3B weights use the **NVIDIA non-commercial licence**, and Gemma uses the Gemma terms.
  This is fine for a hackathon demo, but not for commercial use without other licensing. See `README.md` → Licensing.
- Full technical documentation: `README.md`.
