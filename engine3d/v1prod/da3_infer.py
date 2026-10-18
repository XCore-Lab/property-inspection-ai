"""Frame sampling + Depth Anything 3 inference (chunked for long videos).

DA3-BASE @ 504 px on ~100 frames peaks at ~5.6 GB, the limit of the 6 GB GPU.
Longer tours are processed in overlapping chunks; every chunk is brought into
the frame of the previous one with a similarity transform estimated on the
overlapping frames:
  * scale    = median ratio of the two chunks' depth maps on the overlap frames
  * rotation = chordal mean of the per-frame rotation offsets
  * translation = mean of the per-frame camera-centre offsets
The depth maps of the chunk are multiplied by the same scale, so depth and
poses stay consistent. Overlap frames keep the prediction of the earlier chunk.
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def log(msg):
    print(msg, flush=True)


def extract_frames(video, fps, max_frames=None, max_side=None):
    """Sample the video at ~fps. If max_frames is exceeded, the sampling
    interval is increased so the whole video is still covered. Frames whose longer
    side exceeds max_side are downscaled (memory: 1080p / 4K phone videos)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(vfps / fps))
    if max_frames and n_total // interval > max_frames:
        interval = int(np.ceil(n_total / max_frames))
    frames, idx_list, idx = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            if max_side and max(f.shape[:2]) > max_side:
                sc = max_side / max(f.shape[:2])
                f = cv2.resize(f, (int(round(f.shape[1] * sc)) // 2 * 2, int(round(f.shape[0] * sc)) // 2 * 2),
                               interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            idx_list.append(idx)
        idx += 1
    cap.release()
    return frames, idx_list, vfps, interval


def _to44(e):
    T = np.eye(4)
    T[:3, :4] = e[:3, :4]
    return T


def _infer(model, frames, process_res):
    p = model.inference(image=frames, process_res=process_res,
                        process_res_method="upper_bound_resize")
    return (p.depth.astype(np.float32), p.conf.astype(np.float32),
            np.stack([_to44(e) for e in p.extrinsics]), p.intrinsics.astype(np.float64),
            p.processed_images)


def _chunk_ranges(N, size, overlap):
    if N <= size:
        return [(0, N)]
    out, s = [], 0
    step = size - overlap
    while True:
        e = min(N, s + size)
        out.append((s, e))
        if e == N:
            break
        s += step
    # avoid a tiny last chunk: shift it back so it has a full window
    if len(out) > 1 and out[-1][1] - out[-1][0] < size:
        out[-1] = (max(0, N - size), N)
    return out


def _sim3_from_overlap(dA, eA, dB, eB):
    """Transform chunk B (depth dB, w2c eB) into chunk A's frame, using frames
    present in both. Returns (scale, M) with w2c_A ~= w2c_B' @ inv(M) where
    w2c_B' has translation scaled by `scale`."""
    ok = (dA > 0) & (dB > 0) & np.isfinite(dA) & np.isfinite(dB)
    s = float(np.median(dA[ok] / dB[ok]))
    Rs, ts = [], []
    for TA, TB in zip(eA, eB):
        TB = TB.copy()
        TB[:3, 3] *= s
        M = np.linalg.inv(TA) @ TB             # B-world -> A-world
        Rs.append(M[:3, :3])
        ts.append(M[:3, 3])
    R = Rotation.from_matrix(np.array(Rs)).mean().as_matrix()
    # translation: centres c_A = R c_B + t  ->  t = mean(c_A - R c_B)
    cA = np.array([np.linalg.inv(T)[:3, 3] for T in eA])
    cB = np.array([np.linalg.inv(T)[:3, 3] * s for T in eB])
    t = (cA - cB @ R.T).mean(0)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    res = np.linalg.norm(cA - (cB @ R.T + t), axis=1)
    return s, M, float(res.mean())


def run_da3(frames, model_name="DA3-BASE", process_res=504, chunk_size=100, overlap=20,
            cache=None):
    """Returns dict(depth, conf, w2c (N,4,4), K (N,3,3) at processed res, img, info)."""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if len(d["depth"]) == len(frames):
            log(f"  loaded cached DA3 prediction: {cache}")
            ext = d["ext"]
            w2c = np.stack([_to44(e) for e in ext]) if ext.shape[1] == 3 else ext
            return {"depth": d["depth"].astype(np.float32), "conf": d["conf"].astype(np.float32),
                    "w2c": w2c.astype(np.float64), "K": d["K"].astype(np.float64), "img": d["img"],
                    "info": {"cached": True, "chunks": 1}}
        log(f"  cache {cache} has {len(d['depth'])} frames, need {len(frames)} -> recomputing")
    import torch
    from depth_anything_3.api import DepthAnything3
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("  WARNING: no CUDA GPU - Depth Anything 3 runs on CPU (slow); use a GPU (Cloud Run L4) for videos")
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}").to(device)
    N = len(frames)
    ranges = _chunk_ranges(N, chunk_size, overlap)
    depth = conf = img = None
    w2c = np.zeros((N, 4, 4))
    K = np.zeros((N, 3, 3))
    info = {"cached": False, "chunks": len(ranges), "chunk_ranges": ranges, "links": []}
    done = np.zeros(N, bool)
    for ci, (s, e) in enumerate(ranges):
        dk, ck, ek, Kk, ik = _infer(model, frames[s:e], process_res)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if depth is None:
            depth = np.zeros((N,) + dk.shape[1:], np.float32)
            conf = np.zeros_like(depth)
            img = np.zeros((N,) + ik.shape[1:], ik.dtype)
        if ci > 0:
            ov = [k for k in range(s, e) if done[k]]
            sc, M, res = _sim3_from_overlap(depth[ov], w2c[ov], dk[[k - s for k in ov]], ek[[k - s for k in ov]])
            dk = dk * sc
            ek = ek.copy()
            ek[:, :3, 3] *= sc
            ek = np.stack([T @ np.linalg.inv(M) for T in ek])
            info["links"].append({"chunk": ci, "overlap": len(ov), "scale": sc, "centre_residual": res})
            log(f"  chunk {ci+1}/{len(ranges)} frames {s}-{e-1}: joined on {len(ov)} overlap frames "
                f"(scale x{sc:.3f}, centre residual {res:.3f} DA3 units)")
        else:
            log(f"  chunk 1/{len(ranges)} frames {s}-{e-1}")
        new = [k for k in range(s, e) if not done[k]]
        idx = [k - s for k in new]
        depth[new], conf[new], w2c[new], K[new], img[new] = dk[idx], ck[idx], ek[idx], Kk[idx], ik[idx]
        done[new] = True
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    info["seconds"] = time.time() - t0
    if cache:
        np.savez(cache, depth=depth, conf=conf, ext=w2c, K=K, img=img, t=info["seconds"])
    return {"depth": depth, "conf": conf, "w2c": w2c, "K": K, "img": img, "info": info}
