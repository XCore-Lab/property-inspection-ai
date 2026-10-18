# Deploy runbook (Windows, Git Bash)

From nothing to a public URL. The slowest part is uploading the ~9 GB of model weights once from this PC:
about **2 h at 10 Mbit/s** and about **1 h at 20 Mbit/s**. Start it (step 5) as soon as billing is set up.
After that, every deploy uploads only the code (a few MB).

All commands run in **Git Bash**, from the project folder:

```bash
cd /e/quntbit/Google_Hackathon_v1
```

## 1. Install the tools (~10 min)

1. **Git for Windows** (provides Git Bash): https://git-scm.com/download/win. Skip this if Git Bash is already installed.
2. **Google Cloud CLI**: https://cloud.google.com/sdk/docs/install. Run the Windows installer and keep "bundled Python"
   ticked. It also installs `bq`.
3. Close and reopen Git Bash, then check that both work:
   ```bash
   gcloud --version
   bq version
   ```
   If either says "command not found", see Troubleshooting at the end.

## 2. Sign in

```bash
gcloud auth login
```

A browser window opens. Sign in with the Google account that has the $300 free trial.

## 3. Create the project

Choose a project ID: 6–30 characters, lowercase letters, digits and hyphens, unique across Google Cloud
(for example `pi-hackathon-2026`).

```bash
gcloud projects create pi-hackathon-2026 --name "Property Inspection AI"
gcloud config set project pi-hackathon-2026
```

The rest of this runbook writes `<project-id>` for this ID.

## 4. Billing: link it, upgrade the trial, add a budget alert

1. **Link billing**: Console → Billing → *My projects* → the project → *Change billing* → choose the trial
   account. Or in Git Bash:
   ```bash
   gcloud billing accounts list
   gcloud billing projects link <project-id> --billing-account <ACCOUNT_ID>
   ```
2. **Upgrade to a paid account. GPUs need this**: free-trial billing accounts get no GPU quota.
   Console → Billing → *Overview* → **Activate full account**. The unused part of the $300 credit carries over
   and is spent first.
3. **Budget alert**: Console → Billing → *Budgets & alerts* → *Create budget*. Scope it to this project,
   set the amount to e.g. $250, and add alert thresholds at **$100** and **$200** (actual spend). Alerts only
   send email; they don't stop the service.

Costs (Cloud Run list prices, us-central1): the GPU service (1 L4, 8 vCPU, 32 GiB) costs **about $1.42 per hour**
while an instance is running. That's $0.67 for the L4, $0.52 for the CPUs and $0.23 for the memory.
- `MIN_INSTANCES=1` keeps one instance running all the time: **about $34 per day** (roughly $1,000 a month). The $300
  credit lasts about 8 days at that rate. **Set it back to 0 as soon as judging is over** (see "After judging").
- With `MIN_INSTANCES=0` you pay only while it works. One Lite video costs about **$1.00**, including the ~15 min the
  instance stays up afterwards and the Gemini calls.

## 5. Start the weights upload (the long step)

```bash
gcloud services enable storage.googleapis.com
PROJECT_ID=<project-id> ./deploy/stage_models.sh
```

This uploads `models/` (~9 GB) to `gs://<project-id>-property-inspection-build/models`. If the connection
drops, run the same command again: files that already finished are skipped. Leave it running and open a
second Git Bash window for anything else.

## 6. Deploy

When step 5 has finished (or in the same window, after it):

```bash
PROJECT_ID=<project-id> MIN_INSTANCES=1 ./deploy/deploy.sh
```

The first run builds the base image (CUDA programs, PyTorch and the weights; **~40–60 min** on Cloud Build),
then the app image (~5–10 min), then deploys Cloud Run with one L4 GPU. At the end it prints the URL.

`MIN_INSTANCES=1` keeps the GPU warm for the judges (no cold start), but it bills **about $34 per day** until you
set it back to 0 (see "After judging"). Leave `MIN_INSTANCES=1` out to deploy with 0.

What can go wrong at this step:
- **Cloud Build refuses the 32-vCPU machine** (an error that mentions the machine type or quota, most
  likely while the account is still a free trial): run it again on the smaller machine. The compile is
  slower, roughly 1–2 h.
  ```bash
  PROJECT_ID=<project-id> MIN_INSTANCES=1 BASE_MACHINE=E2_HIGHCPU_8 ./deploy/deploy.sh
  ```
- **"You do not have quota for using GPUs without zonal redundancy"**: Cloud Run normally grants L4 quota
  automatically on a project's first L4 deployment in a region (3 GPUs without zonal redundancy, which is
  what `deploy.sh` asks for), so no quota request is needed up front. If you get this error:
  1. Check Console → IAM & Admin → *Quotas & system limits*. Filter for Cloud Run Admin API,
     "Total Nvidia L4 GPU allocation without zonal redundancy", region `us-central1`.
  2. If it shows 0, try another region. The images stay in us-central1, so there is no rebuild:
     ```bash
     PROJECT_ID=<project-id> MIN_INSTANCES=1 REGION=europe-west4 IMAGE_REGION=us-central1 ./deploy/deploy.sh
     ```
  3. Still 0: ask Google Cloud support to raise it (there is at least one report of a trial account
     upgraded to paid that stayed at 0). Meanwhile, use the CPU-only fallback in step 8.
- **Re-running is safe**: `deploy.sh` skips whatever already exists (the base image, buckets, the service account).

## 7. Check it works

```bash
curl https://<your-service-url>/api/v1/health
```

The response should contain `"cuda": true` and `"gpu": "NVIDIA L4"`. Open the URL in a browser, go to
**Video walkthroughs** and upload `samples/room_walkthrough.mp4` in **Lite** mode. It takes about 10 minutes.

## 8. Fallback: no GPU tonight

A CPU-only service that shows walkthroughs which were already processed on this PC:

```bash
PROJECT_ID=<project-id> GPU=0 ./deploy/deploy.sh
PROJECT_ID=<project-id> ./deploy/upload_walkthroughs.sh room_walkthrough_lite
```

The second command copies `data/walkthrough/room_walkthrough_lite/` (~30 MB) into the bucket that the service
mounts. Reload the page and it is listed; no redeploy is needed. On this service, processing a new video needs
the GPU, so uploads get a "needs the GPU" message.

When the GPU becomes available, run step 6 again without `GPU=0`. It is the same service and the base image
is reused.

## Later deploys (code changes)

```bash
PROJECT_ID=<project-id> MIN_INSTANCES=1 ./deploy/deploy.sh
```

This uploads only the code and takes about 10 minutes. Add `BUILD_BASE=1` only if `models/` or the GPU
programs changed. Every deploy sets the minimum instances again: drop `MIN_INSTANCES=1` once judging is over.

## After judging: stop the costs (do this the same day)

`MIN_INSTANCES=1` costs about $34 per day for as long as it stays on. Set it back to 0 right after the live demo
or judging window:

```bash
gcloud run services update property-inspection-ai --region us-central1 --min-instances 0
```

Use the same `--region` you deployed to. With 0 instances the service scales to zero when idle and costs nothing
while unused; the next visitor waits a few minutes for a cold start, and each Lite video costs about $1.00.
To stop all costs, delete the project: Console → IAM & Admin → *Settings* → *Shut down*.

## Options

- `APP_API_KEY=<secret>`: the page asks for this key once, so the URL isn't open to everyone. If you set it,
  give the key to the judges in the submission.
- `REGION=...`: where the service runs. L4 regions include us-central1, europe-west1, europe-west4 and
  asia-southeast1.

## Troubleshooting

- **`gcloud: command not found` in Git Bash**: add the Cloud SDK to `PATH` for this window. A per-user install
  lives here:
  ```bash
  export PATH="$PATH:/c/Users/$USERNAME/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin"
  ```
  An all-users install is under `/c/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin`.
- **`gcloud` runs in Command Prompt but fails in Git Bash** (Python errors): make Git Bash use the Windows
  launchers. `export -f` also passes them on to the deploy scripts:
  ```bash
  gcloud() { gcloud.cmd "$@"; }; bq() { bq.cmd "$@"; }; export -f gcloud bq
  ```
- **`Permission denied` when running `./deploy/deploy.sh`**: run it as `bash deploy/deploy.sh` with the same
  variables in front.
- **Billing or API errors on the first `gcloud services enable`**: billing isn't linked to the project yet
  (step 4.1).
