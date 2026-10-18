"""Section 2 - Feature tracking with persistent IDs (OpenCV LK optical flow).

Lucas-Kanade is designed for small inter-frame motion, so tracks are followed
through EVERY frame of the original video (not just the DA3-sampled ones);
observations are only recorded at the sampled frames. Each step uses a
forward-backward consistency check, and at every sampled frame the surviving
correspondences are checked against an essential-matrix RANSAC so tracks
that slide along edges or onto moving objects are terminated.
"""

import cv2
import numpy as np

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def normalize_points(pts, K):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, None).reshape(-1, 2)


def essential_inliers(p1, p2, K1, K2, px_thresh=1.0):
    """RANSAC essential-matrix inlier mask for pixel correspondences."""
    if len(p1) < 8:
        return np.ones(len(p1), bool)
    n1, n2 = normalize_points(p1, K1), normalize_points(p2, K2)
    f = 0.5 * (K1[0, 0] + K2[0, 0])
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=method,
                                   prob=0.999, threshold=px_thresh / f)
    if E is None or mask is None:
        return np.ones(len(p1), bool)
    return mask.ravel().astype(bool)


class TrackResult:
    def __init__(self, n_frames):
        self.obs = {}                                   # id -> [(k, u, v)]
        self.frame_obs = [dict() for _ in range(n_frames)]  # k -> {id: (u, v)}
        self.survival = [None] * n_frames
        self.n_active = [0] * n_frames
        self.n_new = [0] * n_frames

    def track_length(self, tid):
        return len(self.obs[tid])

    def age_at(self, tid, k):
        first = self.obs[tid][0][0]
        return k - first


def track_video(video_path, sample_idx, K_full, max_pts=1000, min_pts=600,
                fb_thresh=1.0, min_dist=10, size=None):
    """Run LK through the video. sample_idx[k] = original frame index of sample k.
    size=(W, H): resize every video frame first (must match the sampled frames)."""
    sample_pos = {f: k for k, f in enumerate(sample_idx)}
    last = sample_idx[-1]
    res = TrackResult(len(sample_idx))

    cap = cv2.VideoCapture(video_path)
    prev = None
    pts = np.zeros((0, 2), np.float32)
    ids = np.zeros(0, np.int64)
    next_id = 0
    last_sample_pts = {}
    idx = 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
            frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        if prev is not None and len(pts):
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts.reshape(-1, 1, 2), None, **LK_PARAMS)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **LK_PARAMS)
            p1, p0r = p1.reshape(-1, 2), p0r.reshape(-1, 2)
            fb = np.linalg.norm(p0r - pts, axis=1)
            inside = (p1[:, 0] >= 0) & (p1[:, 0] < W) & (p1[:, 1] >= 0) & (p1[:, 1] < H)
            good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_thresh) & inside
            pts, ids = p1[good].astype(np.float32), ids[good]

        if idx in sample_pos:
            k = sample_pos[idx]
            # Geometric check against the previous sampled observation
            if k > 0 and len(ids):
                common = np.array([i in last_sample_pts for i in ids])
                if common.sum() >= 8:
                    p_prev = np.array([last_sample_pts[i] for i in ids[common]])
                    inl = essential_inliers(p_prev, pts[common], K_full[k - 1], K_full[k])
                    keep = np.ones(len(ids), bool)
                    keep[np.flatnonzero(common)[~inl]] = False
                    pts, ids = pts[keep], ids[keep]
                alive = sum(1 for i in ids if i in last_sample_pts)
                res.survival[k] = alive / max(len(last_sample_pts), 1)

            # Replenish features away from existing tracks
            if len(pts) < min_pts:
                mask = np.full(gray.shape, 255, np.uint8)
                for u, v in pts:
                    cv2.circle(mask, (int(u), int(v)), min_dist, 0, -1)
                new = cv2.goodFeaturesToTrack(gray, maxCorners=max_pts - len(pts),
                                              qualityLevel=0.01, minDistance=min_dist,
                                              mask=mask)
                if new is not None:
                    new = new.reshape(-1, 2).astype(np.float32)
                    new_ids = np.arange(next_id, next_id + len(new))
                    next_id += len(new)
                    pts = np.vstack([pts, new])
                    ids = np.concatenate([ids, new_ids])
                    res.n_new[k] = len(new)

            for tid, (u, v) in zip(ids.tolist(), pts.tolist()):
                res.obs.setdefault(tid, []).append((k, u, v))
                res.frame_obs[k][tid] = (u, v)
            res.n_active[k] = len(ids)
            last_sample_pts = dict(zip(ids.tolist(), pts.tolist()))

        prev = gray
        idx += 1
    cap.release()
    return res
