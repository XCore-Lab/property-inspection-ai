"""Object names by multi-view voting of Gemini boxes.

Gemini (same model / credentials as the inspection app: PI_GEMINI_MODEL, PI_GEMINI_API_KEY or
PI_GEMINI_VERTEX=1 + PI_GCP_PROJECT) returns labelled boxes from a fixed furniture vocabulary on
frames spread over the video. For every object and every detection frame where the object is
visible (projected points inside the image, not occluded by the model depth), each box containing
most of the object's visible points votes for its label, weighted by how well the box fits the
object's projected extent. No SAM masks are produced, so objects are named but not re-split.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

VOCAB = ["sofa", "armchair", "chair", "table", "coffee table", "dining table", "desk", "bed", "cabinet",
         "cupboard", "wardrobe", "shelf", "bookshelf", "drawers", "television", "fan", "lamp", "door",
         "curtain", "refrigerator", "washing machine", "sink", "toilet", "bathtub", "stool", "bench",
         "plant", "mirror", "picture frame", "clock", "air conditioner", "kitchen counter", "stove",
         "staircase", "bucket", "box", "bag"]
SYN = {"armchair": "armchair", "coffee table": "table", "desk": "table", "cupboard": "cabinet",
       "wardrobe": "wardrobe", "bookshelf": "shelf", "drawers": "cabinet", "kitchen counter": "counter"}
N_FRAMES = int(os.environ.get("PI_LABEL_FRAMES", "24"))
SCHEMA = {"type": "object", "required": ["objects"], "properties": {"objects": {"type": "array", "items": {
    "type": "object", "required": ["label", "box_2d"],
    "properties": {"label": {"type": "string", "enum": VOCAB},
                   "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}}}}}}


def log(msg):
    print(msg, flush=True)


def _canon(label, vocab):
    """Longest vocabulary match ('coffee table' before 'table')."""
    best = None
    for v in sorted(vocab, key=len, reverse=True):
        if v in label:
            best = v
            break
    if best is None:
        return None
    return SYN.get(best, best)


def _client():
    from google import genai
    timeout = genai.types.HttpOptions(timeout=90_000)
    if os.environ.get("PI_GEMINI_VERTEX", "").lower() in ("1", "true", "yes", "on"):
        return genai.Client(vertexai=True, http_options=timeout,
                            project=os.environ.get("PI_GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"),
                            location=os.environ.get("PI_GCP_LOCATION", "global"))
    key = os.environ.get("PI_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return genai.Client(api_key=key, http_options=timeout) if key else None


def _detect_frame(client, model, rgb):
    from google.genai import types
    H, W = rgb.shape[:2]
    s = min(1.0, 768 / max(H, W))
    img = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    prompt = ("Detect every clearly visible piece of furniture or household object in this room photo. "
              "Use only these labels: " + ", ".join(VOCAB) + ". Return {\"objects\": [{\"label\", \"box_2d\": "
              "[ymin, xmin, ymax, xmax] normalised to 0-1000}]}, one tight box per object, at most 15.")
    cfg = dict(temperature=0.0, max_output_tokens=1500, response_mime_type="application/json",
               response_json_schema=SCHEMA)
    if "flash" in model:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    for attempt in range(5):
        try:
            r = client.models.generate_content(
                model=model, config=types.GenerateContentConfig(**cfg),
                contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"), prompt])
            items = json.loads(r.text or "{}").get("objects", [])
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 4 or not any(c in str(e) for c in ("429", "500", "503", "UNAVAILABLE", "timed out")):
                log(f"  Gemini object detection failed on a frame: {str(e)[:120]}")
                return []
            import time
            time.sleep(5 * (attempt + 1))
    out = []
    for it in items:
        b = it.get("box_2d") or []
        if len(b) != 4:
            continue
        y0, x0, y1, x1 = (float(v) for v in b)
        out.append({"label": str(it.get("label", "")), "score": 0.9,
                    "box": [min(x0, x1) * W / 1000, min(y0, y1) * H / 1000, max(x0, x1) * W / 1000,
                            max(y0, y1) * H / 1000]})
    return out


def detect(cfg, st, work):
    out = f"{work}/detections.json"
    if os.path.exists(out):
        return json.load(open(out))
    client = _client()
    if client is None:
        log("  object names: no Gemini credentials (PI_GEMINI_API_KEY or PI_GEMINI_VERTEX) - objects stay numbered")
        return None
    model = os.environ.get("PI_GEMINI_MODEL", "gemini-3.8-flash")
    # frames spread over the WHOLE video (not just its beginning)
    u = list(st["usable"])
    use = [u[i] for i in np.linspace(0, len(u) - 1, min(N_FRAMES, len(u))).round().astype(int)]
    log(f"  Gemini ({model}) object detection on {len(use)} frames ...")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda k: _detect_frame(client, model, st["frames"][k]), use))
    dets = {str(k): r for k, r in zip(use, res)}
    n = sum(len(r) for r in res)
    if n == 0:
        log("  object names: Gemini returned no objects - objects stay numbered")
        return None
    data = {"vocab": VOCAB, "detections": dets, "source": model}
    with open(out, "w") as f:
        json.dump(data, f)
    log(f"  {n} labelled boxes")
    return data


def name_objects(cfg, st, objects, work, depth_model=None):
    if not objects:
        return
    res = detect(cfg, st, work)
    if res is None:
        return
    vocab, dets = res["vocab"], res["detections"]
    K, w2c = st["K_full"], st["w2c_model"]
    H, W = st["frames"][0].shape[:2]
    from .render import mesh_depths
    frames_used = sorted(int(k) for k in dets)
    if depth_model is None:
        depth_model = dict(zip(frames_used, mesh_depths(st["model"], K, w2c, W, H, frames=frames_used)))
    counts = {}
    for o in objects:
        P = np.asarray(o["mesh"].vertices)
        if len(P) > 1500:
            P = P[np.random.default_rng(0).choice(len(P), 1500, replace=False)]
        votes, seen = {}, 0
        for k in frames_used:
            Xc = P @ w2c[k][:3, :3].T + w2c[k][:3, 3]
            z = Xc[:, 2]
            ok = z > 0.2
            u = K[k][0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + K[k][0, 2]
            v = K[k][1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + K[k][1, 2]
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if ok.mean() < 0.5:
                continue
            idx = np.flatnonzero(ok)
            dm = depth_model[k][v[idx].astype(int), u[idx].astype(int)]
            vis = idx[(dm == 0) | (z[idx] < dm + 0.06)]
            if len(vis) < 0.4 * len(P):
                continue
            seen += 1
            uu, vv = u[vis], v[vis]
            ob = np.array([uu.min(), vv.min(), uu.max(), vv.max()])
            frame_best = {}
            for d in dets[str(k)] if str(k) in dets else dets.get(k, []):
                lab = _canon(d["label"], vocab)
                if lab is None:
                    continue
                x0, y0, x1, y1 = d["box"]
                inside = np.mean((uu >= x0) & (uu <= x1) & (vv >= y0) & (vv <= y1))
                if inside < 0.6:
                    continue
                ix = max(0, min(x1, ob[2]) - max(x0, ob[0])) * max(0, min(y1, ob[3]) - max(y0, ob[1]))
                ua = (x1 - x0) * (y1 - y0) + (ob[2] - ob[0]) * (ob[3] - ob[1]) - ix
                iou = ix / max(ua, 1e-6)
                if iou < 0.25:
                    continue
                s = d["score"] * iou
                frame_best[lab] = max(frame_best.get(lab, 0), s)
            for lab, s in frame_best.items():
                votes[lab] = votes.get(lab, 0) + s
        o["label_votes"] = {k_: round(v_ / max(seen, 1), 3) for k_, v_ in sorted(votes.items(), key=lambda x: -x[1])}
        o["label_frames"] = seen
        if votes:
            lab, v = max(votes.items(), key=lambda x: x[1])
            if seen >= 2 and v / seen >= 0.08:
                o["label"] = lab
        counts[o["label"]] = counts.get(o["label"], 0) + 1
    log(f"  object names: " + ", ".join(f"{o['id']}={o['label']}" for o in objects))
