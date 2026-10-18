"""Temporal smoothing of per-frame pose corrections.

Frame-to-model ICP corrects every frame independently: the low-frequency part of the
corrections is real (drift / scale fixes), the high-frequency part is per-frame noise
(a hand-held camera moves smoothly). The correction (camera-centre shift + rotation
offset relative to the reference trajectory) is Gaussian-smoothed over time and
re-applied."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def smooth_corrections(w2c_ref, w2c_new, sigma=1.5):
    if sigma <= 0 or len(w2c_ref) < 5:
        return w2c_new
    c2w_r = np.linalg.inv(w2c_ref)
    c2w_n = np.linalg.inv(w2c_new)
    dc = c2w_n[:, :3, 3] - c2w_r[:, :3, 3]
    rv = Rotation.from_matrix(c2w_n[:, :3, :3] @ np.transpose(c2w_r[:, :3, :3], (0, 2, 1))).as_rotvec()
    dc_s = gaussian_filter1d(dc, sigma, axis=0, mode="nearest")
    rv_s = gaussian_filter1d(rv, sigma, axis=0, mode="nearest")
    out = c2w_r.copy()
    out[:, :3, :3] = Rotation.from_rotvec(rv_s).as_matrix() @ c2w_r[:, :3, :3]
    out[:, :3, 3] = c2w_r[:, :3, 3] + dc_s
    return np.linalg.inv(out)
