"""Section 6 - Reconstruction weights + keyframe selection.

Two modes (--recon-mode):
  complete (default): every non-blurred frame, equal TSDF weight 1.
      TSDF weighting also carves: surfaces seen only by low-weight frames get
      erased by higher-weight free-space observations. Where DA3 and OpenCV
      disagree (texture-less walls) that shows up as holes, so the default
      keeps all sharp frames equal and relies on the refined poses + depth
      filtering for quality.
  clean: reliability weights below (spec: lower weight or exclusion).

Weights in 'clean' mode:
  0  excluded : blurred frame (Laplacian blur POOR) or an unconfirmed pose jump
  1  low      : ACCEPTABLE / low-texture frame, or pose LOW-CONF / UNCHECKED
  2  normal   : GOOD frame with a VALID / REFINED pose
Low texture alone never excludes a frame: DA3 depth does not need texture,
and texture-less walls are exactly where OpenCV features run out.

A usable frame becomes a keyframe when the camera moved enough since the last
keyframe (translation / rotation) or the view changed enough (fraction of the
last keyframe's tracks still visible). Keyframes build the dense point cloud
(no near-duplicate points); the TSDF mesh fuses every usable frame with its
weight, because TSDF averaging benefits from redundant views.
"""

from .geometry import c2w_of, rot_angle_deg
from .quality import GOOD
from .refine import LOW_CONF, UNCHECKED


def reconstruction_weights(quality, pose_state, jump, mode="complete"):
    if mode == "complete":
        return [0 if q["grades"]["blur"] == 0 else 1 for q in quality]
    w = []
    for q, s, j in zip(quality, pose_state, jump):
        if q["grades"]["blur"] == 0 or (j and s == LOW_CONF):
            w.append(0)
        elif q["quality"] != GOOD or s in (LOW_CONF, UNCHECKED):
            w.append(1)
        else:
            w.append(2)
    return w


def select_keyframes(w2c, tracks, weights, min_trans=0.2, min_rot=10.0, min_overlap=0.7):
    N = len(w2c)
    kfs, reasons = [], {}
    last = None
    for k in range(N):
        if weights[k] == 0:
            continue
        if last is None:
            kfs.append(k)
            reasons[k] = "first"
            last = k
            continue
        A, B = c2w_of(w2c[last]), c2w_of(w2c[k])
        trans = float(((A[:3, 3] - B[:3, 3]) ** 2).sum() ** 0.5)
        rot = rot_angle_deg(A[:3, :3].T @ B[:3, :3])
        ids_last = set(tracks.frame_obs[last])
        overlap = (len(ids_last & set(tracks.frame_obs[k])) / len(ids_last)
                   if ids_last else 0.0)
        why = []
        if trans > min_trans:
            why.append(f"trans {trans:.2f}m")
        if rot > min_rot:
            why.append(f"rot {rot:.0f}deg")
        if overlap < min_overlap:
            why.append(f"overlap {overlap*100:.0f}%")
        if why:
            kfs.append(k)
            reasons[k] = ", ".join(why)
            last = k
    return kfs, reasons
