"""Section 10 - Explainable SLAM diagnostic video (OpenCV).

Layout (3 x 2 panels + live statistics strip):
  [1 tracked features + IDs] [2 DA3 depth + obstacles] [3 RANSAC correspondences]
  [4 trajectory DA3 vs refined] [5 occupancy grid] [6 A* + safety field]
  [------------------------ live statistics ------------------------]
"""

import cv2
import numpy as np

from .obstacles import DANGER_RGB
from .occupancy import cells_px, draw_path, draw_polyline, render_grid, scale_bar
from .quality import ACCEPTABLE, GOOD
from .refine import LOW_CONF, REFINED, UNCHECKED
from .verify import FAIL, INSUFFICIENT, PASS, WARN

PW, PH = 560, 315
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_RGB = {PASS: (60, 220, 60), WARN: (255, 190, 0), FAIL: (255, 60, 60),
              INSUFFICIENT: (170, 170, 170), "N/A": (170, 170, 170)}
POSE_RGB = {"VALID": (60, 220, 60), REFINED: (80, 200, 255), LOW_CONF: (255, 60, 60),
            UNCHECKED: (170, 170, 170)}
QUAL_RGB = {GOOD: (60, 220, 60), ACCEPTABLE: (255, 190, 0), "POOR": (255, 60, 60)}


def text(img, s, org, scale=0.45, color=(255, 255, 255), th=1):
    # Outline by offsetting (OpenCV 5's text engine widens thicker strokes,
    # so a thick black under-stroke would leave ghost glyphs).
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(img, s, (x + dx, y + dy), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, th, cv2.LINE_AA)


def fit(img, W=PW, H=PH, bg=(30, 30, 30)):
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((H, W, 3), bg, np.uint8)
    y, x = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def title(img, s):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(img, s, (8, 16), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)


class DiagnosticRenderer:
    def __init__(self, ctx):
        self.c = ctx
        g = ctx.grid
        self.px = 4
        self.base_grid = cv2.cvtColor(render_grid(g, self.px), cv2.COLOR_BGR2RGB)
        self.base_safe = cv2.cvtColor(render_grid(g, self.px, safety=True,
                                                  planner=ctx.planner), cv2.COLOR_BGR2RGB)
        dim = (self.base_grid * 0.35).astype(np.uint8)
        self.base_traj = dim
        self.raw_px = cells_px(g, ctx.ground.to_2d(ctx.raw_centers), self.px)
        self.ref_px = cells_px(g, ctx.ground.to_2d(ctx.ref_centers), self.px)
        self.headings = [ctx.ground.dir_2d(np.linalg.inv(T)[:3, 2]) for T in ctx.w2c_ref]
        self.lm_ids = set(ctx.lm_by_id)
        # longest verified tracks get their ID printed
        by_len = sorted(ctx.lm_by_id.values(), key=lambda lm: -len(lm["obs"]))
        self.show_ids = {lm["id"] for lm in by_len[:400]}

    # -- panels -----------------------------------------------------------
    def p_tracks(self, k):
        c = self.c
        img = c.frames_full[k].copy()
        H, W = img.shape[:2]
        n_lm = 0
        labelled = 0
        for tid, (u, v) in c.tracks.frame_obs[k].items():
            is_lm = tid in self.lm_ids
            n_lm += is_lm
            obs = c.tracks.obs[tid]
            hist = [(int(x), int(y)) for kk, x, y in obs if k - 3 <= kk <= k]
            col = (0, 255, 120) if is_lm else (255, 210, 0)
            if is_lm:
                for a, b in zip(hist[:-1], hist[1:]):
                    cv2.line(img, a, b, col, 2, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), 4 if is_lm else 2, col, -1, cv2.LINE_AA)
            if is_lm and tid in self.show_ids and labelled < 20 and len(hist) >= 3:
                text(img, f"#{tid}", (int(u) + 5, int(v) - 5), 0.5, (255, 255, 255))
                labelled += 1
        out = fit(img)
        title(out, "1  Tracked features + persistent landmark IDs")
        text(out, f"green = verified landmark ({n_lm})   yellow = track ({len(c.tracks.frame_obs[k])})",
             (8, PH - 10), 0.42)
        return out

    def p_depth(self, k):
        c = self.c
        vis, info = c.detector.render(c.depth[k], c.K_proc[k], c.w2c_ref[k], c.valid[k])
        out = fit(vis)
        title(out, "2  DA3 depth (filtered) + obstacle zones")
        text(out, "green floor | red <1m | orange <2m | yellow <4m", (8, PH - 10), 0.42)
        self._obst = info
        return out

    def p_matches(self, k):
        c = self.c
        out = np.full((PH, PW, 3), 30, np.uint8)
        title(out, "3  SIFT matches -> essential-matrix RANSAC")
        if k == 0 or c.ver[k] is None:
            text(out, "first frame: no pair", (20, PH // 2))
            return out
        v = c.ver[k]
        w2 = PW // 2
        A = cv2.resize(c.frames_full[k - 1], (w2, int(w2 * c.frames_full[0].shape[0] / c.frames_full[0].shape[1])))
        B = cv2.resize(c.frames_full[k], A.shape[1::-1])
        y0 = 30
        out[y0:y0 + A.shape[0], :w2] = A
        out[y0:y0 + B.shape[0], w2:w2 * 2] = B
        s = w2 / c.frames_full[0].shape[1]
        if len(v["pts_a"]):
            idx = np.arange(len(v["pts_a"]))
            rng = np.random.default_rng(k)
            if len(idx) > 160:
                idx = rng.choice(idx, 160, replace=False)
            for t in idx:
                inl = v["inlier_mask"][t] if len(v["inlier_mask"]) else False
                p = (int(v["pts_a"][t][0] * s), int(v["pts_a"][t][1] * s) + y0)
                q = (int(v["pts_b"][t][0] * s) + w2, int(v["pts_b"][t][1] * s) + y0)
                cv2.line(out, p, q, (60, 230, 60) if inl else (230, 50, 50), 1, cv2.LINE_AA)
        yb = y0 + A.shape[0] + 22
        text(out, f"frame {k} -> {k+1}   matches {v['matches']}   inliers {v['inliers']} "
             f"({v['inlier_ratio']*100:.1f}%)", (8, yb), 0.45)
        re = "n/a" if not np.isfinite(v["rot_err"]) else f"{v['rot_err']:.2f} deg"
        te = ("low parallax" if v["low_parallax"] else f"{v['trans_err']:.1f} deg")
        text(out, f"DA3 vs OpenCV: rotation err {re} | translation-dir err {te}", (8, yb + 22), 0.45)
        text(out, f"parallax {v['parallax_px']:.1f}px   check: {v['status']}", (8, yb + 44), 0.45,
             STATUS_RGB[v["status"]])
        return out

    def p_traj(self, k):
        c = self.c
        img = self.base_traj.copy()
        draw_polyline(img, self.raw_px, (150, 150, 150), 2)
        draw_polyline(img, self.ref_px, (0, 200, 255), 3, upto=k)
        for lp in c.loops:
            cv2.line(img, self.ref_px[lp["i"]], self.ref_px[lp["j"]], (255, 0, 255), 2, cv2.LINE_AA)
        for i in range(k + 1):
            if c.pose_state[i] == LOW_CONF:
                p = self.ref_px[i]
                cv2.drawMarker(img, p, (255, 60, 60), cv2.MARKER_TILTED_CROSS, 12, 2)
        for i in c.keyframes:
            if i <= k:
                cv2.circle(img, self.ref_px[i], 3, (255, 255, 255), -1)
        self._marker(img, k)
        out = fit(img)
        title(out, "4  Trajectory: DA3 raw (grey) vs verified/refined (cyan)")
        text(out, "white dot = keyframe   x = low-confidence   magenta = loop", (8, PH - 10), 0.42)
        return out

    def _marker(self, img, k):
        p = self.ref_px[k]
        cv2.circle(img, p, 7, (255, 255, 255), -1)
        cv2.circle(img, p, 7, (255, 140, 0), 2)
        h = self.headings[k]
        n = np.linalg.norm(h)
        if n > 1e-6:
            e = (int(p[0] + h[0] / n * 22), int(p[1] - h[1] / n * 22))
            cv2.arrowedLine(img, p, e, (255, 140, 0), 2, tipLength=0.35)

    def p_grid(self, k):
        img = self.base_grid.copy()
        draw_polyline(img, self.ref_px, (40, 170, 220), 2, upto=k)
        self._marker(img, k)
        scale_bar(img, self.c.grid, self.px)
        out = fit(img)
        title(out, "5  Occupancy grid (morphology-refined)")
        text(out, "white free | red obstacle | grey unknown", (8, PH - 10), 0.42)
        return out

    def p_path(self, k):
        c = self.c
        img = self.base_safe.copy()
        path = c.path
        if path:
            tmp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            draw_path(tmp, c.grid, path, self.px, c.planner)
            img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        self._marker(img, k)
        out = fit(img)
        title(out, "6  A* on distance field + safety margin")
        if path:
            text(out, f"path {c.path_len:.2f} m | min clearance {c.clearance[0]:.2f} m | "
                 f"robot r {c.planner.robot_radius:.2f} m", (8, PH - 10), 0.42)
        return out

    # -- statistics strip ------------------------------------------------
    def stats(self, k):
        c = self.c
        strip = np.full((140, PW * 3, 3), 18, np.uint8)
        v = c.ver[k] if k > 0 else None
        st = v["status"] if v else "N/A"
        q = c.quality[k]
        kf = k in c.keyframes
        obst = self._obst
        d = obst["min_distance"]
        rm = c.frame_rmse[k]
        loop = [lp for lp in c.loops if lp["j"] == k]
        cols = [
            [("DA3 Pose", c.pose_state[k], POSE_RGB[c.pose_state[k]]),
             ("OpenCV Geometric Check", st, STATUS_RGB[st]),
             ("Frame Quality", q["quality"], QUAL_RGB[q["quality"]]),
             ("Keyframe", ("YES (" + c.kf_reasons[k] + ")") if kf else "NO",
              (60, 220, 60) if kf else (170, 170, 170))],
            [("SIFT Features", f"{c.n_feats[k]:,}", None),
             ("Matches", f"{v['matches']:,}" if v else "-", None),
             ("RANSAC Inliers", f"{v['inliers']:,}" if v else "-", None),
             ("Inlier Ratio", f"{v['inlier_ratio']*100:.1f}%" if v else "-", None)],
            [("Reprojection RMSE", f"{rm:.2f} px" if np.isfinite(rm) else "-", None),
             ("Tracked Landmarks", f"{sum(t in self.lm_ids for t in c.tracks.frame_obs[k]):,}"
              f" / {len(c.tracks.frame_obs[k]):,} tracks", None),
             ("Obstacle Distance", f"{obst['level']}" + (f" ({d:.2f} m)" if np.isfinite(d) else ""),
              DANGER_RGB[obst["level"]]),
             ("Loop", f"with frame {loop[0]['i']+1} ({loop[0]['inliers']} inl.)" if loop else "-",
              (255, 0, 255) if loop else None)],
        ]
        for ci, col in enumerate(cols):
            x = 16 + ci * PW
            for ri, (lab, val, color) in enumerate(col):
                y = 28 + ri * 30
                cv2.putText(strip, lab + ":", (x, y), FONT, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
                cv2.putText(strip, val, (x + 215, y), FONT, 0.55, color or (255, 255, 255),
                            1 + (color is not None), cv2.LINE_AA)
        text(strip, f"Frame {k+1}/{len(c.w2c_ref)}", (PW * 3 - 140, 130), 0.5)
        return strip

    def frame(self, k):
        top = np.hstack([self.p_tracks(k), self.p_depth(k), self.p_matches(k)])
        bot = np.hstack([self.p_traj(k), self.p_grid(k), self.p_path(k)])
        grid = np.vstack([top, bot])
        cv2.line(grid, (PW, 0), (PW, 2 * PH), (60, 60, 60), 2)
        cv2.line(grid, (2 * PW, 0), (2 * PW, 2 * PH), (60, 60, 60), 2)
        cv2.line(grid, (0, PH), (3 * PW, PH), (60, 60, 60), 2)
        return np.vstack([grid, self.stats(k)])


def write_diagnostic_video(ctx, path, fps):
    r = DiagnosticRenderer(ctx)
    first = r.frame(0)
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (first.shape[1], first.shape[0]))
    stills = {}
    for k in range(len(ctx.w2c_ref)):
        f = first if k == 0 else r.frame(k)
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        if k in (len(ctx.w2c_ref) // 3, len(ctx.w2c_ref) * 2 // 3):
            stills[k] = f
    wr.release()
    return stills
