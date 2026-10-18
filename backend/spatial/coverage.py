"""Inspection coverage: which parts of the walls / floor / ceiling the camera has seen,
and seen well enough to inspect, over time.

Surfaces are sampled every `step` metres on the clean layout (walls from layout.json,
floor + ceiling over the room rectangles, floor under furniture excluded). For every
SLAM frame each sample is projected into the camera and checked against that frame's
depth map (so furniture in front of a wall occludes it). Per sample we keep:

  first_seen    first SLAM frame that saw it at all (<= max_range)            -> "seen"
  first_good    first frame that saw it inspectably (<= good_range, not too
                oblique, frame not BAD)                                        -> "inspectable"
  best_dist     closest distance it was seen from
  inspected     True if one of the AI-inspected keyframes saw it inspectably  -> "inspected"
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.spatial.slam_data import SlamRun

SURF_WALL, SURF_FLOOR, SURF_CEIL = 0, 1, 2


@dataclass
class Samples:
    xyz: np.ndarray            # (N,3)
    normal: np.ndarray         # (N,3) facing the room
    kind: np.ndarray           # (N,) SURF_*
    group: list[str]           # (N,) wall id / room id (floor, ceiling)
    t: np.ndarray              # (N,) position along the wall (walls) else 0
    step: float
    groups: dict = field(default_factory=dict)   # group id -> metadata (label, kind, length/area)


def _in_footprint(xz: np.ndarray, fp: dict, pad: float = 0.05) -> np.ndarray:
    c = np.asarray(fp["center"], float)
    sx, sz = np.asarray(fp["size"], float) / 2 + pad
    a = np.deg2rad(fp.get("angle_deg", 0.0))
    d = xz - c
    u = d[:, 0] * np.cos(a) + d[:, 1] * np.sin(a)
    v = -d[:, 0] * np.sin(a) + d[:, 1] * np.cos(a)
    return (np.abs(u) <= sx) & (np.abs(v) <= sz)


_SIDE = {("x", 1): "west", ("x", -1): "east", ("z", 1): "north", ("z", -1): "south"}


def make_samples(layout: dict, step: float = 0.1) -> Samples:
    ceil = float(layout.get("ceiling_height_m", 2.6))
    P, N, Kd, G, T = [], [], [], [], []
    groups: dict = {}
    for w in layout["walls"]:
        ts = np.arange(w["t0"] + step / 2, w["t1"], step)
        ys = np.arange(step / 2, ceil, step)
        tt, yy = np.meshgrid(ts, ys, indexing="ij")
        tt, yy = tt.ravel(), yy.ravel()
        if w["axis"] == "x":
            p = np.c_[np.full_like(tt, w["offset"]), yy, tt]
            n = np.array([w["sign"], 0, 0], float)
        else:
            p = np.c_[tt, yy, np.full_like(tt, w["offset"])]
            n = np.array([0, 0, w["sign"]], float)
        P.append(p); N.append(np.tile(n, (len(p), 1))); Kd.append(np.full(len(p), SURF_WALL))  # noqa: E702
        G += [w["id"]] * len(p); T.append(tt)  # noqa: E702
        # plan-view side label: the wall's normal points INTO the room, so a wall whose
        # normal points +z (south in the plan, which has -z up) is the room's north wall
        groups[w["id"]] = {"kind": "wall", "length_m": round(w["t1"] - w["t0"], 2),
                           "label": f"{_SIDE.get((w['axis'], int(w['sign'])), '')} wall ({w['id']})".strip()}
    objs = [o for o in layout.get("objects", []) if not o.get("ceiling_mounted") and o.get("footprint")]
    rooms = layout.get("rooms", [])
    fm = layout.get("_floor_mask")
    if fm is not None:
        # true floor region (Production's floor raster: row = z, col = x), resampled to `step`
        fl, lo, cell = fm["floor"], fm["lo"], fm["cell"]
        rows, cols = np.nonzero(fl)
        xs = np.arange(lo[0] + cols.min() * cell, lo[0] + (cols.max() + 1) * cell, step) + step / 2
        zs = np.arange(lo[1] + rows.min() * cell, lo[1] + (rows.max() + 1) * cell, step) + step / 2
        xx, zz = np.meshgrid(xs, zs, indexing="ij")
        xz = np.c_[xx.ravel(), zz.ravel()]
        ci = np.clip(((xz[:, 0] - lo[0]) / cell).astype(int), 0, fl.shape[1] - 1)
        ri = np.clip(((xz[:, 1] - lo[1]) / cell).astype(int), 0, fl.shape[0] - 1)
        xz = xz[fl[ri, ci]]
    else:                                   # fallback: room rectangles
        parts = []
        for r in rooms:
            x0, z0, x1, z1 = r["bbox_m"]
            xx, zz = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(z0 + step / 2, z1, step), indexing="ij")
            parts.append(np.c_[xx.ravel(), zz.ravel()])
        xz = np.vstack(parts) if parts else np.zeros((0, 2))
    # assign each floor cell to the room whose rectangle contains it, else "other"
    rid = np.full(len(xz), "", object)
    for r in rooms:
        x0, z0, x1, z1 = r["bbox_m"]
        m = (rid == "") & (xz[:, 0] >= x0 - 0.05) & (xz[:, 0] <= x1 + 0.05) & (xz[:, 1] >= z0 - 0.05) & (xz[:, 1] <= z1 + 0.05)
        rid[m] = r["id"]
    rid[rid == ""] = "other"
    names = {r["id"]: r["name"] for r in rooms}
    names["other"] = "Other area"
    under = np.zeros(len(xz), bool)
    for o in objs:
        under |= _in_footprint(xz, o["footprint"])
    for g in [r["id"] for r in rooms] + ["other"]:
        m = rid == g
        if not m.any():
            continue
        fxz, cxz = xz[m & ~under], xz[m]
        P.append(np.c_[fxz[:, 0], np.zeros(len(fxz)), fxz[:, 1]]); N.append(np.tile([0, 1, 0], (len(fxz), 1)))
        Kd.append(np.full(len(fxz), SURF_FLOOR)); G += [g] * len(fxz); T.append(np.zeros(len(fxz)))  # noqa: E702
        P.append(np.c_[cxz[:, 0], np.full(len(cxz), ceil), cxz[:, 1]]); N.append(np.tile([0, -1, 0], (len(cxz), 1)))
        Kd.append(np.full(len(cxz), SURF_CEIL)); G += [g + "_ceiling"] * len(cxz); T.append(np.zeros(len(cxz)))  # noqa: E702
        groups[g] = {"kind": "floor", "label": f"{names[g]} floor", "area_m2": round(len(fxz) * step * step, 2),
                     "furniture_excluded_m2": round(int((m & under).sum()) * step * step, 2)}
        groups[g + "_ceiling"] = {"kind": "ceiling", "label": f"{names[g]} ceiling",
                                  "area_m2": round(len(cxz) * step * step, 2)}
    return Samples(xyz=np.vstack(P), normal=np.vstack(N).astype(float), kind=np.concatenate(Kd), group=G,
                   t=np.concatenate(T), step=step, groups=groups)


@dataclass
class Coverage:
    samples: Samples
    first_seen: np.ndarray      # (N,) frame index or -1
    first_good: np.ndarray      # (N,) frame index or -1
    best_dist: np.ndarray       # (N,) metres (inf if unseen)
    good_by_frame: list         # per frame: indices of samples inspectable in that frame
    seen_by_frame: list         # per frame: indices of samples seen (any quality) in that frame
    inspected: np.ndarray       # (N,) bool - seen inspectably by an AI-inspected keyframe
    params: dict

    def state(self, upto: int | None = None) -> np.ndarray:
        """0 unseen, 1 seen (far/oblique), 2 inspectable, 3 inspected by AI; `upto` = frame index (live)."""
        n = len(self.first_seen)
        s = np.zeros(n, np.int8)
        lim = np.inf if upto is None else upto
        s[(self.first_seen >= 0) & (self.first_seen <= lim)] = 1
        s[(self.first_good >= 0) & (self.first_good <= lim)] = 2
        if upto is None:
            s[self.inspected] = 3
        return s

    def stats(self, upto: int | None = None) -> dict:
        st = self.state(upto)
        S = self.samples
        out = {"by_kind": {}, "by_group": {}}
        for k, name in ((SURF_WALL, "walls"), (SURF_FLOOR, "floor"), (SURF_CEIL, "ceiling")):
            m = S.kind == k
            tot = max(1, int(m.sum()))
            out["by_kind"][name] = {"seen_pct": round(100 * float((st[m] >= 1).sum()) / tot, 1),
                                    "inspectable_pct": round(100 * float((st[m] >= 2).sum()) / tot, 1),
                                    "inspected_pct": round(100 * float((st[m] >= 3).sum()) / tot, 1),
                                    "area_m2": round(tot * S.step * S.step, 1)}
        G = np.array(S.group)
        for g, meta in S.groups.items():
            m = G == g
            tot = max(1, int(m.sum()))
            out["by_group"][g] = {**meta, "seen_pct": round(100 * float((st[m] >= 1).sum()) / tot, 1),
                                  "inspectable_pct": round(100 * float((st[m] >= 2).sum()) / tot, 1),
                                  "inspected_pct": round(100 * float((st[m] >= 3).sum()) / tot, 1)}
        allm = np.ones(len(st), bool)
        tot = len(st)
        out["overall"] = {"seen_pct": round(100 * float((st >= 1).sum()) / tot, 1),
                          "inspectable_pct": round(100 * float((st >= 2).sum()) / tot, 1),
                          "inspected_pct": round(100 * float((st[allm] >= 3).sum()) / tot, 1)}
        return out

    def gaps(self, min_len_m: float = 0.4) -> list[dict]:
        """Wall stretches never seen inspectably (all heights), longest first - 'go back and film this'."""
        S = self.samples
        st = self.state()
        G = np.array(S.group)
        out = []
        for g, meta in S.groups.items():
            if meta["kind"] != "wall":
                continue
            m = np.where(G == g)[0]
            ts = np.round(S.t[m] / S.step).astype(int)
            cols = {}
            for i, tk in zip(m, ts):
                cols.setdefault(tk, []).append(st[i] >= 2)
            keys = sorted(cols)
            bad = [k for k in keys if np.mean(cols[k]) < 0.5]
            run = []
            for k in keys + [None]:
                if k is not None and k in bad and (not run or k == run[-1] + 1):
                    run.append(k)
                    continue
                if run and len(run) * S.step >= min_len_m:
                    out.append({"surface": g, "label": meta["label"], "from_t": round(run[0] * S.step, 2),
                                "to_t": round((run[-1] + 1) * S.step, 2), "length_m": round(len(run) * S.step, 2)})
                run = [k] if (k is not None and k in bad) else []
        for g, meta in S.groups.items():
            if meta["kind"] in ("floor", "ceiling"):
                m = G == g
                frac = float((st[m] >= 2).mean()) if m.any() else 1
                if frac < 0.8:
                    out.append({"surface": g, "label": meta["label"], "length_m": None,
                                "uncovered_m2": round(float((st[m] < 2).sum()) * S.step ** 2, 1)})
        return sorted(out, key=lambda x: -(x.get("length_m") or x.get("uncovered_m2") or 0))


def compute_coverage(run: SlamRun, samples: Samples, max_range: float = 6.0, good_range: float = 3.0,
                     max_angle_deg: float = 70.0, margin_px: int = 4) -> Coverage:
    X = samples.xyz
    n = len(X)
    first_seen = np.full(n, -1, int)
    first_good = np.full(n, -1, int)
    best = np.full(n, np.inf)
    good_by_frame = []
    seen_by_frame = []
    cos_max = np.cos(np.deg2rad(max_angle_deg))
    h, w = run.depth.shape[1:]
    Xh = np.c_[X, np.ones(n)]
    for f in range(run.n):
        w2c = np.linalg.inv(run.c2w[f])
        pc = (Xh @ w2c.T)[:, :3]
        z = pc[:, 2]
        K = run.K_depth[f]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        ok = (z > 0.2) & (u >= margin_px) & (u < w - margin_px) & (v >= margin_px) & (v < h - margin_px)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            good_by_frame.append(np.zeros(0, int))
            seen_by_frame.append(np.zeros(0, int))
            continue
        ui, vi = u[idx].astype(int), v[idx].astype(int)
        d = run.depth[f][vi, ui]
        zz = z[idx]
        tol = np.maximum(0.12, 0.06 * zz)
        # visible = depth map does not show something clearly in front of the sample;
        # invalid depth (0) counts as not visible
        vis = (d > 0) & (d >= zz - tol)
        idx = idx[vis]
        C = run.c2w[f][:3, 3]
        ray = X[idx] - C
        dist = np.linalg.norm(ray, axis=1)
        cosang = np.abs(np.sum(-ray / dist[:, None] * samples.normal[idx], axis=1))
        seen = idx[dist <= max_range]
        new = seen[first_seen[seen] < 0]
        first_seen[new] = f
        best[idx] = np.minimum(best[idx], dist)
        good_ok = (dist <= good_range) & (cosang >= cos_max) & (run.frame_quality[f] != "BAD")
        good = idx[good_ok]
        newg = good[first_good[good] < 0]
        first_good[newg] = f
        good_by_frame.append(good)
        seen_by_frame.append(seen)
    return Coverage(samples=samples, first_seen=first_seen, first_good=first_good, best_dist=best,
                    good_by_frame=good_by_frame, seen_by_frame=seen_by_frame, inspected=np.zeros(n, bool),
                    params={"max_range_m": max_range, "good_range_m": good_range, "max_angle_deg": max_angle_deg,
                            "sample_step_m": samples.step})


def select_keyframes(cov: Coverage, sharpness: np.ndarray, max_frames: int = 16, min_frames: int = 6,
                     min_gain_frac: float = 0.005, min_gap: int = 3, seen_weight: float = 0.25,
                     weights: dict | None = None) -> list[int]:
    """Greedy set cover: each pick maximises newly covered surface x sharpness, where a
    sample counts 1.0 if the frame sees it inspectably and `seen_weight` if it only sees
    it (far / oblique) - so large rooms walked through the middle still get inspected.
    At least `min_frames` keyframes are returned (topped up with the sharpest frame of
    each remaining time slot) so the AI always inspects the whole walkthrough."""
    weights = weights or {SURF_WALL: 1.0, SURF_FLOOR: 1.0, SURF_CEIL: 0.6}
    wv = np.vectorize(lambda k: weights.get(int(k), 1.0))(cov.samples.kind)
    total = float(wv.sum())
    got = np.zeros(len(wv))                   # best credit reached so far per sample (0, seen_weight, 1)
    sharp = np.clip(sharpness / (np.percentile(sharpness, 90) + 1e-6), 0.2, 1.0)
    chosen: list[int] = []
    n = len(cov.good_by_frame)
    while len(chosen) < max_frames:
        best, bf = 0.0, -1
        for f in range(n):
            if any(abs(f - c) < min_gap for c in chosen):
                continue
            g, s = cov.good_by_frame[f], cov.seen_by_frame[f]
            gain = float((wv[g] * (1.0 - got[g])).sum() + (wv[s] * np.clip(seen_weight - got[s], 0, None)).sum())
            gain *= sharp[f]
            if gain > best:
                best, bf = gain, f
        if bf < 0 or best < min_gain_frac * total:
            break
        chosen.append(bf)
        got[cov.seen_by_frame[bf]] = np.maximum(got[cov.seen_by_frame[bf]], seen_weight)
        got[cov.good_by_frame[bf]] = 1.0
    if len(chosen) < min(min_frames, n):     # top up: sharpest frame of each uncovered time slot
        for sl in np.array_split(np.arange(n), min(min_frames, n)):
            if len(chosen) >= min_frames:
                break
            if any(sl[0] - min_gap < c <= sl[-1] + min_gap for c in chosen):
                continue
            chosen.append(int(sl[np.argmax(sharpness[sl])]))
    return sorted(chosen)


def mark_inspected(cov: Coverage, frames: list[int]) -> None:
    for f in frames:
        cov.inspected[cov.good_by_frame[f]] = True
