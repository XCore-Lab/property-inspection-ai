"""Section 1 - OpenCV 5 camera pose verification.

For consecutive frames: SIFT (or ORB) features -> Lowe ratio test ->
findEssentialMat (USAC-MAGSAC / RANSAC) with the DA3 intrinsics ->
recoverPose. The OpenCV relative rotation and translation DIRECTION are then
compared with the relative transform between the two DA3 poses.
"""

import cv2
import numpy as np

from .geometry import relative, rot_angle_deg, vec_angle_deg
from .tracking import normalize_points

PASS, WARN, FAIL, INSUFFICIENT = "PASS", "WARN", "FAIL", "INSUFFICIENT"
_RANK = {PASS: 0, WARN: 1, INSUFFICIENT: 2, FAIL: 3}


def make_detector(kind="sift", n=2000):
    if kind == "orb":
        return cv2.ORB_create(nfeatures=n), cv2.NORM_HAMMING
    return cv2.SIFT_create(nfeatures=n), cv2.NORM_L2


def detect_all(grays, kind="sift", n=2000):
    det, norm = make_detector(kind, n)
    out = []
    for g in grays:
        kp, desc = det.detectAndCompute(g, None)
        pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
        out.append((pts, desc))
    return out, norm


def ratio_match(d1, d2, norm, ratio=0.75):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [(m.queryIdx, m.trainIdx) for m, *rest in
            (p for p in knn if len(p) == 2) if m.distance < ratio * rest[0].distance]
    if not good:
        return np.zeros((0, 2), int)
    # mutual check: keep a match only if it is also the best match backwards
    back = {m.queryIdx: m.trainIdx for m in bf.match(d2, d1)}
    good = [(q, t) for q, t in good if back.get(t) == q]
    return np.array(good, int).reshape(-1, 2)


def spatial_coverage(pts, shape, n=4):
    """Fraction of an n x n image grid touched by the points (0..1)."""
    if len(pts) == 0:
        return 0.0
    H, W = shape
    gx = np.clip((pts[:, 0] / W * n).astype(int), 0, n - 1)
    gy = np.clip((pts[:, 1] / H * n).astype(int), 0, n - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / (n * n)


def homography_ratio(pA, pB, e_inliers, px=2.0):
    """Share of the essential-matrix inliers also explained by one homography.
    ~1.0 means planar scene or pure rotation -> translation is ill-posed."""
    if e_inliers.sum() < 8:
        return 1.0
    H, m = cv2.findHomography(pA[e_inliers], pB[e_inliers], cv2.RANSAC, px)
    return 0.0 if m is None else float(m.mean())


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson_px(n1, n2, R, t, f):
    """Sampson epipolar error (pixels) of normalized correspondences under (R, t)."""
    E = skew(t / max(np.linalg.norm(t), 1e-12)) @ R
    x1 = np.hstack([n1, np.ones((len(n1), 1))])
    x2 = np.hstack([n2, np.ones((len(n2), 1))])
    Ex1 = x1 @ E.T
    Etx2 = x2 @ E
    num = np.sum(x2 * Ex1, axis=1) ** 2
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-18)) * f


def rotation_parallax_px(n1, n2, R, f):
    """Median residual motion after removing rotation (translation-induced parallax)."""
    r = np.hstack([n1, np.ones((len(n1), 1))]) @ R.T
    proj = r[:, :2] / r[:, 2:3]
    return float(np.median(np.linalg.norm(proj - n2, axis=1)) * f)


def verify_pair(featA, featB, K_A, K_B, w2c_A, w2c_B, norm,
                rot_thresh=(1.5, 4.0), trans_thresh=(15.0, 35.0),
                min_parallax_px=15.0, px_thresh=1.0, img_shape=(480, 848),
                planar_ratio=0.9, ratio=0.75):
    ptsA, dA = featA
    ptsB, dB = featB
    m = ratio_match(dA, dB, norm, ratio)
    res = {"features_a": len(ptsA), "features_b": len(ptsB), "matches": len(m),
           "inliers": 0, "inlier_ratio": 0.0, "rot_err": np.nan, "trans_err": np.nan,
           "parallax_px": 0.0, "low_parallax": True, "status": INSUFFICIENT,
           "pts_a": np.zeros((0, 2)), "pts_b": np.zeros((0, 2)),
           "inlier_mask": np.zeros(0, bool), "R_cv": None, "t_cv": None,
           "da3_epi_inlier_ratio": np.nan, "h_ratio": np.nan, "coverage": 0.0,
           "degenerate": False}
    if len(m) < 30:
        return res

    pA, pB = ptsA[m[:, 0]], ptsB[m[:, 1]]
    nA, nB = normalize_points(pA, K_A), normalize_points(pB, K_B)
    f = 0.5 * (K_A[0, 0] + K_B[0, 0])
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    E, mask = cv2.findEssentialMat(nA, nB, np.eye(3), method=method,
                                   prob=0.999, threshold=px_thresh / f)
    res.update(pts_a=pA, pts_b=pB)
    if E is None or E.shape != (3, 3):
        return res
    _, R, t, mask_pose = cv2.recoverPose(E, nA, nB, np.eye(3), mask=mask.copy())
    inl = mask_pose.ravel() > 0
    t = t.ravel()

    T = relative(w2c_A, w2c_B)
    R_da3, t_da3 = T[:3, :3], T[:3, 3]
    parallax = rotation_parallax_px(nA[inl], nB[inl], R, f) if inl.any() else 0.0
    low_par = parallax < min_parallax_px

    rot_err = rot_angle_deg(R.T @ R_da3)
    trans_err = vec_angle_deg(t, t_da3)
    da3_epi = sampson_px(nA, nB, R_da3, t_da3, f) < 2.0 * px_thresh
    res.update(inliers=int(inl.sum()), inlier_ratio=float(inl.mean()),
               R_cv=R, t_cv=t, rot_err=rot_err,
               trans_err=trans_err if not low_par else np.nan,
               parallax_px=parallax, low_parallax=low_par, inlier_mask=inl,
               da3_epi_inlier_ratio=float(da3_epi.mean()))

    h_ratio = homography_ratio(pA, pB, inl)
    cov = min(spatial_coverage(pA[inl], img_shape), spatial_coverage(pB[inl], img_shape))
    # planar scene (one wall) with real parallax: E is ill-conditioned
    degenerate = h_ratio > planar_ratio and not low_par
    res.update(h_ratio=h_ratio, coverage=cov, degenerate=degenerate)

    if res["inliers"] < 20 or res["inlier_ratio"] < 0.3 or cov < 0.25:
        res["status"] = INSUFFICIENT
        return res
    s_rot = PASS if rot_err < rot_thresh[0] else WARN if rot_err < rot_thresh[1] else FAIL
    if low_par:
        s_tr = PASS
    else:
        s_tr = (PASS if trans_err < trans_thresh[0] else
                WARN if trans_err < trans_thresh[1] else FAIL)
    res["status"] = max(s_rot, s_tr, key=_RANK.get)
    if degenerate and res["status"] == FAIL:
        res["status"] = INSUFFICIENT     # can't blame DA3 on degenerate geometry
    return res


def verify_sequence(feats, K_full, w2c, norm, img_shape=(480, 848)):
    """Consecutive-pair verification. Result k compares frame k-1 -> k (k>=1)."""
    out = [None]
    for k in range(1, len(feats)):
        out.append(verify_pair(feats[k - 1], feats[k], K_full[k - 1], K_full[k],
                               w2c[k - 1], w2c[k], norm, img_shape=img_shape))
    return out


def pose_agreement(ver, K_full, w2c):
    """Re-score existing correspondences against a (possibly refined) pose set."""
    rot, trans, epi = [], [], []
    for k in range(1, len(w2c)):
        v = ver[k]
        if v is None or v["R_cv"] is None or v["status"] == INSUFFICIENT:
            continue
        T = relative(w2c[k - 1], w2c[k])
        rot.append(rot_angle_deg(v["R_cv"].T @ T[:3, :3]))
        if not v["low_parallax"]:
            trans.append(vec_angle_deg(v["t_cv"], T[:3, 3]))
        nA = normalize_points(v["pts_a"], K_full[k - 1])
        nB = normalize_points(v["pts_b"], K_full[k])
        f = 0.5 * (K_full[k - 1][0, 0] + K_full[k][0, 0])
        e = sampson_px(nA[v["inlier_mask"]], nB[v["inlier_mask"]], T[:3, :3], T[:3, 3], f)
        epi.append(float(np.mean(e < 2.0)))
    return {"rot_err_mean": float(np.mean(rot)) if rot else np.nan,
            "trans_err_median": float(np.median(trans)) if trans else np.nan,
            "epipolar_inlier_ratio": float(np.mean(epi)) if epi else np.nan}
