"""Section 3 - Trajectory refinement and stabilization.

Policy (per consecutive pair, from the OpenCV verification):
  PASS         -> DA3 and OpenCV agree: keep DA3 (high confidence)
  WARN         -> small disagreement: OpenCV relative rotation / translation
                  direction used as a soft constraint (refinement)
  FAIL         -> large disagreement: pose marked LOW-CONFIDENCE, NOT changed
  INSUFFICIENT -> no reliable geometry: keep DA3, marked unchecked
Sudden jumps (robust z-score on step length / rotation) that OpenCV does not
confirm are also marked low-confidence. A light second-difference prior
stabilises jitter; low-confidence poses are pinned to DA3.

OpenCV's essential matrix has no scale, so translation MAGNITUDES always come
from DA3; only directions are refined.
"""

import numpy as np

from .geometry import average_rotations, c2w_of, rot_angle_deg
from .verify import FAIL, INSUFFICIENT, PASS, WARN

VALID, REFINED, LOW_CONF, UNCHECKED = "VALID", "REFINED", "LOW-CONF", "UNCHECKED"


def _robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med)) + 1e-9
    return (x - med) / mad


def refine_trajectory(w2c_raw, ver, smooth=0.5, w_minor=4.0, jump_z=3.5):
    N = len(w2c_raw)
    c2w = np.stack([c2w_of(T) for T in w2c_raw])
    R0, P0 = c2w[:, :3, :3], c2w[:, :3, 3]

    steps = np.r_[0, np.linalg.norm(np.diff(P0, axis=0), axis=1)]
    rsteps = np.r_[0, [rot_angle_deg(R0[k - 1].T @ R0[k]) for k in range(1, N)]]
    z_t = _robust_z(steps[1:]) if N > 2 else np.zeros(N - 1)
    z_r = _robust_z(rsteps[1:]) if N > 2 else np.zeros(N - 1)
    jump = np.r_[False, (z_t > jump_z) | (z_r > jump_z)]

    status = [None] + [v["status"] for v in ver[1:]]
    state = [VALID] * N
    for k in range(1, N):
        s = status[k]
        if s == FAIL or (jump[k] and s != PASS):
            state[k] = LOW_CONF
        elif s == INSUFFICIENT:
            state[k] = UNCHECKED
        elif s == WARN:
            state[k] = REFINED
    minor = [k >= 1 and status[k] == WARN and state[k] != LOW_CONF for k in range(N)]

    # --- Rotations: blend DA3 absolute with OpenCV-chained neighbours (WARN pairs)
    R = R0.copy()
    for _ in range(10):
        for k in range(N):
            if state[k] == LOW_CONF:
                continue
            Rs, ws = [R0[k]], [1.0]
            if minor[k]:
                Rs.append(R[k - 1] @ ver[k]["R_cv"].T)
                ws.append(1.0)
            if k + 1 < N and minor[k + 1]:
                Rs.append(R[k + 1] @ ver[k + 1]["R_cv"])
                ws.append(1.0)
            if len(Rs) > 1:
                R[k] = average_rotations(Rs, ws)

    # --- Positions: linear least squares
    rows, rhs, wts = [], [], []

    def add(coeffs, b, w):
        r = np.zeros(N)
        for idx, c in coeffs:
            r[idx] = c
        rows.append(r)
        rhs.append(b)
        wts.append(w)

    for k in range(N):
        add([(k, 1.0)], P0[k], 100.0 if state[k] == LOW_CONF else 1.0)
    for k in range(1, N):
        v = ver[k]
        d_da3 = P0[k] - P0[k - 1]
        if minor[k] and not v["low_parallax"] and v["t_cv"] is not None:
            cv_dir = -(R[k] @ v["t_cv"])            # C_k - C_{k-1} in world
            cv_dir /= max(np.linalg.norm(cv_dir), 1e-12)
            L = np.linalg.norm(d_da3)
            blend = d_da3 / max(L, 1e-12) + cv_dir
            d = blend / max(np.linalg.norm(blend), 1e-12) * L
            add([(k, 1.0), (k - 1, -1.0)], d, w_minor)
        elif status[k] == PASS:
            add([(k, 1.0), (k - 1, -1.0)], d_da3, 1.0)
    if smooth > 0:
        for k in range(1, N - 1):
            if LOW_CONF in (state[k - 1], state[k], state[k + 1]):
                continue
            add([(k - 1, 1.0), (k, -2.0), (k + 1, 1.0)], np.zeros(3), smooth)

    A = np.array(rows) * np.sqrt(wts)[:, None]
    B = np.array(rhs) * np.sqrt(wts)[:, None]
    P = np.linalg.lstsq(A, B, rcond=None)[0]

    w2c_ref = np.zeros_like(w2c_raw)
    for k in range(N):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R[k], P[k]
        w2c_ref[k] = np.linalg.inv(T)

    info = {
        "state": state, "jump": jump.tolist(), "steps": steps, "rot_steps": rsteps,
        "pos_correction": np.linalg.norm(P - P0, axis=1),
        "rot_correction": np.array([rot_angle_deg(R0[k].T @ R[k]) for k in range(N)]),
    }
    return w2c_ref, info


def smoothness_metrics(w2c):
    c2w = np.stack([c2w_of(T) for T in w2c])
    P, R = c2w[:, :3, 3], c2w[:, :3, :3]
    acc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=1)
    rel = [R[k - 1].T @ R[k] for k in range(1, len(R))]
    rot_acc = [rot_angle_deg(rel[k - 1].T @ rel[k]) for k in range(1, len(rel))]
    return {"pos_accel_mean": float(acc.mean()), "rot_accel_mean": float(np.mean(rot_acc)),
            "path_length": float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))}
