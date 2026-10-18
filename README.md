# Property Inspection AI — risk & valuation on Google Cloud

**AI property inspection for insurers, mortgage lenders and real-estate valuers.**
Upload a phone walkthrough video of a property and get, in minutes:

* every visible defect, localized and verified (cracks, water damage, mold, broken glass, rust, …)
* a **condition grade (A–E)**, **risk level**, **insurance perils**, **indicative repair-cost range**,
  **valuation / collateral impact**, **premium-loading suggestion** and an **underwriting decision** with reasons
* **evidence-integrity (fraud) signals**: the same photo reused for another property, edited photos,
  photos of a screen / renderings / AI-generated images, old photos
* for videos: a **3D model of the property**, a **2D floor plan**, and a **coverage map** that proves which
  walls / floors / ceilings were actually inspected (and which must be filmed again)
* a **Gemini underwriting assistant** that answers questions about the inspection, citing issue ids
* a **portfolio view** over every inspection record (JSON behind the REST API; optional BigQuery / Pub/Sub hand-off)

Built by **Team XCore** for the Google Cloud AI Builder Cup 2026 (BFSI track). Pitch deck and documentation: [`docs/`](docs/README.md).

## See it in action

The included sample, `samples/room_walkthrough.mp4` (a 20-second phone video of a living room and corridor), processed
in Lite mode on a laptop RTX 3080 in about 10.5 minutes: 197 frames, 15 AI keyframes, 7 findings, 50.6% of surfaces
inspectable, condition grade **A (95.7 / 100)**. The decision is **REFER: additional evidence required**, because too
little of the property was filmed close enough to inspect, and the report lists exactly which areas to film again.

**1. Upload and live scan.** While a video is processed, the progress view replays it under a scanning beam and shows
each pipeline step with an ETA.

<img src="docs/media/scanning-view.jpg" alt="Processing view: the walkthrough video under a scanning beam, with progress and the current pipeline step" width="760">

**2. Detection on the AI keyframes.** LocateAnything-3B proposes defect boxes, Gemma 4 E4B checks a full-resolution
close-up of each box, and the rule engine marks it *Detected* or *Needs review* (amber).

<table>
  <tr>
    <td width="50%"><img src="docs/media/detection-door-window.jpg" alt="Keyframe with two boxes: door surface damage and window surface damage, both Needs review"></td>
    <td width="50%"><img src="docs/media/detection-wall-mold.jpg" alt="Keyframe with one box: visible mold on the wall, Needs review"></td>
  </tr>
  <tr>
    <td>Door and window frame: surface damage <img src="docs/media/closeup-window.jpg" alt="Close-up of the window frame" height="96" align="right"></td>
    <td>Wall: possible mold patch near the floor <img src="docs/media/closeup-wall-mold.jpg" alt="Close-up of the wall patch" height="96" align="right"></td>
  </tr>
</table>

**3. Live replay.** `live_map.mp4` replays the walk: the video on the left, and on the right the floor plan filling in as
the camera moves (amber = seen, green = inspectable within 3 m, purple = AI keyframes), with coverage bars over time.

![Live replay: video, coverage bars and the floor plan filling in as the camera moves](docs/media/live-replay.gif)

[Full-quality replay video (MP4)](docs/media/live-replay.mp4)

**4. Final coverage map.** Every 10 cm of wall, floor and ceiling is checked against every frame. The map shows the walk
path, the AI keyframes, the numbered findings, and the areas that must be filmed again.

<table>
  <tr>
    <td width="45%"><img src="docs/media/coverage-map.png" alt="Final coverage map with walk path, keyframes, numbered findings and coverage statistics"></td>
    <td>
      <b>Coverage</b><br>
      Walls: seen 78%, inspectable 65.5%<br>
      Floor: seen 74%, inspectable 69.3%<br>
      Ceiling: seen 40.6%, inspectable 8.4%<br><br>
      <b>Film again</b><br>
      Main room ceiling (19.8 m²) and floor (3.6 m²)<br>
      Corridor ceiling (3.5 m²) and floor (2.1 m²)<br>
      South wall (1.7 m) and east wall (1.5 m)
    </td>
  </tr>
</table>

**5. 3D report.** The interactive 3D model with numbered finding pins (click one to fly to it), the findings list,
property risk and valuation, and the Gemini property summary and underwriting assistant.

<img src="docs/media/report-3d.jpg" alt="3D walkthrough report: 3D model with finding pins, findings list, risk card and Gemini summary">

**Dashboard.** Condition grades, risk levels, underwriting decisions and repair exposure across all walkthroughs.

<img src="docs/media/dashboard.png" alt="Dashboard with walkthrough figures, recent walkthroughs and condition grade chart">

**Version 1.0.0 (frozen 2026-10-16).** Hybrid AI: **Gemini 3.8 Flash** (scene + summary)
+ two small local GPU models (**LocateAnything-3B** boxes, **Gemma 4 E4B** verifier) + **Depth Anything 3** for 3D.

| Step | Model | Runs on |
|---|---|---|
| Room, components, photo authenticity | Gemini 3.8 Flash | Gemini API (or Vertex AI) |
| Defect boxes | LocateAnything-3B (GGUF, locate-anything.cpp) | GPU (Cloud Run L4 Job) |
| First defect check | Gemma 4 E4B (GGUF, llama.cpp server) | GPU (Cloud Run L4 Job) |
| Rule engine: DETECTED / NEEDS_REVIEW / NOT_DETECTED | deterministic (`fusion_service.py`) | CPU |
| Risk, grade, cost, valuation, decision | deterministic (`risk_rules.yaml`) | CPU |
| Video depth + 3D | Depth Anything 3 (DA3-BASE) + OpenCV 5 SLAM + Open3D TSDF | GPU + CPU (Cloud Run L4 Job) |
| 3D object names, underwriter summary, Q&A | Gemini 3.8 Flash | Gemini API |

### Video modes: Lite and Advanced

Both modes run **the same AI models**, so defects, risk decision, coverage and 3D pin
positions are the same. Only the 3D build differs:

| | **Lite** | **Advanced** (default) |
|---|---|---|
| 3D surface (TSDF voxel) | 3 cm | 1.5 cm |
| Alignment passes | 1 | 2 |
| Preview videos (orbit, depth, SLAM diagnostic) | skipped | yes |
| Frames used / camera tracking | all / identical | all |
| 3D build, 20 s video, RTX 3050 laptop | **11.1 min** | 26.9 min |
| Model-vs-video match (PSNR) | 21.8 dB | 21.6 dB |
| Clean GLB size | 11 MB | 28 MB |

Choose it in the upload form, `mode=lite|advanced` on `POST /api/v1/walkthrough`, or
`scripts/inspect_walkthrough.py --mode lite`. A Lite run is stored as `<video>_lite`. Halving the frame rate
was tested and rejected: tracking broke and the room came out stretched.

## Problem → solution (BFSI track)

| Industry pain | What the app does |
|---|---|
| Insurers price and settle claims on self-reported photos; inspections are slow and costly | Automated defect detection + insurance perils + premium-loading suggestion + claim-ready report |
| Claims / valuation fraud: recycled or edited photos, photos of other properties | Perceptual-hash duplicate detection across all properties, EXIF edit and age checks, Gemini authenticity check → **REFER - evidence integrity** |
| Mortgage lenders need the collateral condition before lending | Condition grade, collateral risk, valuation impact %, repair retention suggestion |
| Remote inspections can't show what was covered | Video → 3D model + coverage map; low AI-inspected coverage → **REFER - additional evidence required** |
| Compliance / explainability of automated decisions | Deterministic rule engine and risk rules (`backend/config/risk_rules.yaml`); every decision lists its reasons; raw model outputs are archived; humans make the final decision |
| Portfolio risk analytics | Every photo / walkthrough is a JSON record behind the REST API (`/api/v1/portfolio`, `/api/v1/walkthroughs`); optional BigQuery rows and Pub/Sub events for downstream underwriting / claims systems (`PI_BQ_DATASET`, `PI_PUBSUB_TOPIC`; off in the deployed setup) |

## How it works

```
photo ─► quality check + integrity (dHash duplicate, EXIF) ─► Gemini: room + components + authenticity
      ─► prompt router (defect_classes.yaml) ─► LocateAnything-3B: defect boxes ─► NMS / merge
      ─► full-resolution crops ─► Gemma 4 E4B verification + CV line/blotch evidence
      ─► rule engine: DETECTED / NEEDS_REVIEW / MANUAL_CHECK_REQUIRED / NOT_DETECTED / UNKNOWN
      ─► risk engine: grade, perils, repair cost, valuation impact, premium loading, decision
      ─► JSON + annotated image + HTML report in the data dir (a Cloud Storage bucket on Cloud Run; BigQuery / Pub/Sub optional)

video ─► 3D engine (engine3d/): Depth Anything 3 depth + OpenCV SLAM poses ─► TSDF fusion ─► walls / floor /
         ceiling / doors ─► objects named by Gemini ─► clean + SLAM GLB models, floor plan, occupancy map
      ─► coverage: every 10 cm of surface checked against every frame's depth (seen / inspectable / AI-inspected)
      ─► AI keyframes = greedy set cover of the inspectable surface ─► photo pipeline (above), one batch:
         each local model is loaded once (stage-major: scene -> boxes -> verify)
      ─► findings placed in 3D ─► property-level risk (coverage = evidence completeness) ─► 3D viewer
```

### Accuracy (30-image pilot set, weak image-level labels)

| Pipeline | Precision / Recall / **F1** | Missed defects | Clean rooms flagged |
|---|---|---|---|
| Gemini scene + Gemini boxes + strict verification (all-Gemini) | 0.44 / 0.84 / **0.58** | 0 | 2/11 |
| LocateAnything-3B boxes (local GPU) + Gemini scene + verification | 0.47 / 0.37 / 0.41 | 26 | 2/11 |
| LocateAnything-3B + Gemma 4 E4B (no Gemini) | 0.57 / 0.42 / 0.48 | 31 | 0/11 |

The shipped hybrid (Gemini scene + LocateAnything + Gemma) has not yet been scored on this set.
On videos it found defects where all-Gemini boxes found almost none. A Gemini second opinion on each finding
is available (`PI_VALIDATE_BACKEND=gemini`) but is off by default. Set `PI_LOCATE_BACKEND=gemini PI_VERIFY_BACKEND=gemini PI_VALIDATE_BACKEND=off`
for the all-Gemini pipeline.

Timing on the RTX 3050 laptop: a photo takes about 30-60 s. For video, the 14 keyframes take about 6 min,
LocateAnything takes ~5 s per frame and Gemma ~3-4 s per box. Gemini costs about US$0.005 per call,
so roughly US$0.35 per walkthrough.

## Google Cloud technologies

| Technology | Use |
|---|---|
| **Cloud Run (service)** | the web UI + API: CPU only, request-based billing, scales to zero |
| **Cloud Run (job, NVIDIA L4)** | one execution per uploaded video: Depth Anything 3 + SLAM, LocateAnything-3B and Gemma 4 on the GPU; billed only while a video is processing |
| **Gemini 3.8 Flash (Gemini API)** | scene + component recognition, image authenticity, 3D object naming, underwriter summary, grounded Q&A (Vertex AI also supported: `PI_GEMINI_VERTEX=1`) |
| **Cloud Storage** | the data dir, mounted on both the service and the job: uploads, job status, reports, 3D models, maps |
| **Secret Manager** | the Gemini API key (and the optional UI password), injected at runtime; no keys in the image or the repo |
| **Cloud Build + Artifact Registry** | builds the ~12 GB image (CUDA binaries, PyTorch, baked-in weights); the weights are fetched from Google Drive inside the build |
| **IAM service account** | least-privilege runtime account: the bucket, the secrets, and starting the GPU job |
| BigQuery, Pub/Sub | optional: `property_risk.inspections` rows and `inspection.completed` / `walkthrough.completed` events (`PI_BQ_DATASET`, `PI_PUBSUB_TOPIC`) |

## Deploy to Google Cloud (low cost)

Two Cloud Run resources share one container image and one Cloud Storage bucket:

```
browser ──► Cloud Run SERVICE  property-inspection-ai   (1 vCPU, 2 GiB, min 0)   ── web UI, uploads, progress, results
                 │  starts one execution per video (Cloud Run Jobs API)
                 ▼
            Cloud Run JOB      property-inspection-gpu  (8 vCPU, 32 GiB, 1× L4)  ── 3D engine + AI inspection
                 │
            Cloud Storage bucket mounted at /app/data on both: uploads/, walkthrough/<name>/, job status
```

Idle cost is the image storage only (≈ US$1.20/month); a 20-minute walkthrough costs ≈ US$0.50 of L4 time plus
≈ US$0.35 of Gemini calls. The UI stays within the Cloud Run free tier.

Requirements: a project with billing, `gcloud` logged in, and a Gemini API key (AI Studio). New projects get L4 quota
automatically in `us-east4`, `europe-west4`, `europe-west1` and `asia-southeast1`; `us-central1` needs a quota request
first, so the job can live in a different region than the service (`JOB_REGION`).

```bash
export CLOUDSDK_CORE_PROJECT=<project-id>
gcloud services enable secretmanager.googleapis.com
printf '%s' "$GEMINI_KEY" | gcloud secrets create gemini-api-key --replication-policy=automatic --data-file=-

JOB_REGION=us-east4 PROJECT_ID=<project-id> DRIVE_ID=<Google_Hackathon_v1.zip file id> ./deploy/deploy_lowcost.sh
```

The first run (≈ 25 min) fetches the weights from Google Drive inside Cloud Build (`deploy/cloudbuild-fetch-models.yaml`,
nothing goes through your laptop), builds the base image (`Dockerfile.base`: CUDA builds of locate-anything.cpp and
llama.cpp, PyTorch, weights) and the app image, then deploys the job and the service and prints the URL. Later runs
rebuild only the code layer (≈ 8 min). Options: `APP_API_KEY_SECRET=<secret>` puts the UI behind a password
(`X-API-Key`, also required for `/walkthroughs/`), `JOB_CPU=4 JOB_MEM=16Gi` for the cheapest L4 size, `GPU_ZR=1` for
zonal-redundant GPUs.

The service is public without an IAM binding (`--no-invoker-iam-check`), which also works in organisations with the
Domain Restricted Sharing policy. Useful commands:

```bash
gcloud run jobs executions list --job property-inspection-gpu --region us-east4      # one row per video
gcloud run services logs read property-inspection-ai --region us-central1 --limit 50
```

`deploy/deploy.sh` is the original all-in-one variant (one GPU service, Vertex AI, BigQuery, Pub/Sub).

## Run it on another PC (teammates)

The repository holds the code only. To run the app on a new machine you need four things:

| What | Where it comes from |
|---|---|
| Access to this private repository | an XCore-Lab admin adds you (Settings → Collaborators, or an org invite) |
| `models\` (~9 GB) and `bin\` (~2 GB) | `Google_Hackathon_v1.zip` from the team (too large for Git) |
| A Gemini API key | your own from AI Studio, or one sent to you privately (never in a group chat or a commit) |
| A Windows PC with an NVIDIA GPU (6 GB+), Python 3.11, Git and ~25 GB free disk | your machine |

Steps:

```bat
git clone https://github.com/XCore-Lab/property-inspection-ai.git
cd property-inspection-ai
:: copy the models\ and bin\ folders from Google_Hackathon_v1.zip into this folder
setup_local.bat          :: once: installs everything (PyTorch is a ~2.5 GB download) and creates .env
:: open .env and put your key after PI_GEMINI_API_KEY=
run_local.bat            :: http://127.0.0.1:8080
```

- **No Gemini key yet?** Add these two lines to `.env` and the app runs fully on the local GPU, with Gemma 4 E4B doing
  the Gemini roles (3D object names then stay numbered):
  ```
  PI_SCENE_BACKEND=gemma
  PI_VLM_BACKEND=gemma
  ```
- **No NVIDIA GPU?** The web app still opens, but videos can't be processed ("Video processing needs the GPU").
- **The dashboard starts empty.** Upload `samples\room_walkthrough.mp4` under **Video walkthroughs** in Lite mode to
  get the demo walkthrough (about 10 minutes on a laptop RTX GPU).
- `.env` holds your key and is ignored by Git; never commit it.

## Run locally (Windows, NVIDIA GPU optional)

```bat
setup_local.bat                         :: creates .venv (PyTorch CUDA, OpenCV, Open3D, FastAPI, google-genai) + .env
:: put your Gemini key in .env:  PI_GEMINI_API_KEY=...   (AI Studio key with billing; free tier = 20 requests/day)
run_local.bat                           :: http://127.0.0.1:8080  (API docs: /docs)
```

Everything is in this folder: `models/locate_anything/locate-anything-q4_k-vitf16.gguf`,
`models/gemma4/gemma-4-E4B-it-Q4_0.gguf` + `mmproj-gemma-4-E4B-it-Q8_0.gguf`, `models/hf/` (Depth Anything 3
BASE), and the Windows GPU programs `bin/locate-anything/locate_anything.dll` and `bin/llama.cpp/llama-server.exe`.
`run_local.bat` points `HF_HOME` at `models\hf`, so the Depth Anything weights load offline.

Linux / macOS: `./run_local.sh`. Without a key, `set PI_MOCK_MODELS=1` runs the whole UI with fake models.

The command line works too:

```bat
.venv\Scripts\python.exe scripts\inspect_folder.py samples\images --property P001
.venv\Scripts\python.exe scripts\inspect_walkthrough.py --video samples\room_walkthrough.mp4 --run-slam --mode lite --property DEMO-001
.venv\Scripts\python.exe -m pytest -q tests          :: 46 tests, mock models (no key / GPU needed)
```

Without a GPU, use the all-Gemini pipeline (see Accuracy) for photos. Depth Anything 3 also runs on CPU, but it is very slow.

## Configuration (`PI_*` env vars or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `PI_GEMINI_API_KEY` | — | AI Studio key (local runs) |
| `PI_GEMINI_VERTEX`, `PI_GCP_PROJECT`, `PI_GCP_LOCATION` | 0, —, global | use Vertex AI instead of an AI Studio key |
| `PI_GEMINI_MODEL` | gemini-3.8-flash | any Gemini model with vision + JSON output |
| `PI_SCENE_BACKEND`, `PI_LOCATE_BACKEND`, `PI_VERIFY_BACKEND`, `PI_VALIDATE_BACKEND` | gemini, capi, gemma, off | model per role (`capi`/`cli` = LocateAnything, `gemma`, `gemini`, `off`, `mock`) |
| `PI_MODELS_DIR`, `PI_LOCATE_LIB`, `PI_LLAMA_SERVER` | models/, bin/... | local model files and binaries |
| `PI_GPU_RESIDENCY` | swap | `swap` = one local model on the GPU at a time (6-8 GB), `coexist` = both loaded (24 GB+ GPUs) |
| `PI_VERIFY_STRICT` | auto | strict verification prompt (on automatically when the verifier is Gemini) |
| `PI_GPU_JOB` | — | Cloud Run job that processes walkthroughs (`projects/P/locations/R/jobs/NAME`); empty = run them in this process on the local GPU |
| `PI_GCS_BUCKET`, `PI_BQ_DATASET`, `PI_PUBSUB_TOPIC` | — | optional Google Cloud integrations (off when empty) |
| `PI_API_KEY` | — | require `X-API-Key` on `/api/*` and `/walkthroughs/` (the UI asks for it once) |
| `PI_CURRENCY`, `PI_COST_MULTIPLIER` | USD, 1.0 | repair-cost currency and regional price index |
| `PI_DA3_MODEL` | DA3-BASE | `DA3-SMALL` = faster 3D, slightly less detail |
| `PI_SLAM_FPS`, `PI_SLAM_MAX_FRAMES` | 10, 600 | video sampling for the 3D engine |
| `PI_MOCK_MODELS` | 0 | fake models (UI / tests without a key) |
| `PI_PHOTO_UPLOAD` | 0 | allow single-photo uploads (`POST /api/v1/inspect`); walkthroughs use the photo pipeline on their keyframes either way |

You can edit risk scoring, repair costs, valuation bands and premium loadings in `backend/config/risk_rules.yaml`, and the defect taxonomy in `backend/config/defect_classes.yaml`.

## API (OpenAPI at `/docs`)

| Method | Path | |
|---|---|---|
| POST | `/api/v1/inspect` | **off by default** (video-only product; `PI_PHOTO_UPLOAD=1` turns it on): multipart `image`, optional `property_id`, `property_value`, `room_type`, `component`, `tiled`; `?wait=true` |
| GET | `/api/v1/inspection/{id}` | full record incl. `risk`, `integrity`, detections, progress |
| GET | `/api/v1/inspection/{id}/report?format=html\|md\|txt\|json` | report |
| POST | `/api/v1/inspection/{id}/summary` | Gemini underwriter summary |
| POST | `/api/v1/inspection/{id}/ask` | `{"question": ...}` grounded Q&A |
| POST | `/api/v1/walkthrough` | multipart `video`, `property_id`, `mode` (`lite` / `advanced`) → 3D + coverage + inspection + risk |
| GET | `/api/v1/walkthroughs`, `/walkthroughs/{name}/` | list, 3D viewer |
| POST | `/api/v1/walkthrough/{name}/ask` | Q&A over a whole property walkthrough |
| GET | `/api/v1/portfolio` | portfolio risk overview |
| GET | `/api/v1/property/{property_id}/report` | multi-image property report with property-level risk |
| GET | `/api/v1/health` | model, GPU and Google Cloud integration status |

## Folder map

| Path | What |
|---|---|
| `backend/services/pipeline.py` | photo pipeline orchestration (parallel) |
| `backend/services/vlm_service.py` | Gemini client, Gemma (llama.cpp server), scene / verification / second-opinion prompts |
| `backend/services/locate_service.py` | LocateAnything-3B (C API / CLI) and Gemini localizers |
| `backend/services/gpu_service.py` | GPU arbiter: loads / swaps the local models, one 3D job at a time |
| `backend/services/fusion_service.py` | NMS, rule engine, safety wording |
| `backend/services/risk_service.py` + `config/risk_rules.yaml` | risk & valuation engine |
| `backend/services/integrity_service.py` | fraud / evidence-integrity signals |
| `backend/services/cloud_service.py` | optional Cloud Storage archive, BigQuery, Pub/Sub |
| `backend/jobs/walkthrough_job.py` | Cloud Run job entry point: processes one walkthrough (`python -m backend.jobs.walkthrough_job --name X`) |
| `deploy/` | `deploy_lowcost.sh` (service + L4 job), Cloud Build configs, Drive weight fetch, original `deploy.sh` |
| `backend/api/assistant.py` | Gemini underwriting assistant + portfolio |
| `backend/spatial/` | coverage map, 3D finding placement, GLB building |
| `scripts/inspect_walkthrough.py` | video → 3D engine → coverage → keyframes → inspection → risk |
| `engine3d/` | 3D engine: Depth Anything 3 + OpenCV SLAM + room model + floor plan |
| `depth_anything_3/` | Depth Anything 3 inference code (weights download from Hugging Face / baked into the image) |
| `frontend/` | web UI (no build step) |
| `samples/` | sample photos (see `images/SOURCES.json`) + a sample walkthrough video |
| `docs/` | documentation: pitch decks (`docs/pitch-deck/`), README media (`docs/media/`), [index](docs/README.md) |

## Limitations and responsible use

- This is AI-assisted visual screening. It does not establish structural safety, root cause, or whether locks, fans or windows work; those are always listed as manual checks.
- Repair costs and valuation effects are **indicative ranges from configurable rules**, not quotes or appraisals.
- Decisions are **recommendations**; a qualified surveyor or underwriter decides.
- Pilot accuracy comes from 30 images with image-level labels. Validate on your own labelled portfolio before relying on it.
- The 3D scale is approximate: it assumes the camera was held about 1.5 m above the floor (`--camera-height` in `PI_SLAM_EXTRA_ARGS`).

## Licensing

- App code: © 2026 Team XCore: Thanveer Ahamed, Janitha Prathapa and Dinal Fernando.
- Depth Anything 3 (DA3-SMALL / DA3-BASE): Apache-2.0.
- Gemini: Google Cloud terms.
- Gemma 4: Gemma terms of use.
- **LocateAnything-3B weights: NVIDIA non-commercial licence.** They are fine for the competition demo and research.
  A commercial product needs another localizer, such as `PI_LOCATE_BACKEND=gemini`.
- Sample photos: see `samples/images/SOURCES.json`.
