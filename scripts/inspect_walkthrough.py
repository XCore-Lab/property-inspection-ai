"""Walkthrough inspection = property inspection + SLAM coverage mapping.

    python scripts/inspect_walkthrough.py --video samples/room_walkthrough.mp4 --run-slam

1. 3D engine (engine3d/run_production.py, same environment, run as a subprocess): Depth Anything 3
   depth + SLAM camera pose for every sampled frame, clean room layout (walls, doors, rooms, objects
   named by Gemini), clean + SLAM GLBs, floor plan. An existing run is reused (--slam-dir, or found automatically).
2. Coverage: every 10 cm of wall / floor / ceiling is checked against every frame's depth
   (occlusion-aware) -> seen / inspectable (<= 3 m, not grazing) over time.
3. AI keyframes chosen by greedy set cover of the inspectable surface, then inspected
   by the normal pipeline (Gemini scene -> LocateAnything boxes -> Gemma verification -> Gemini second opinion).
4. Findings placed in 3D (box -> depth -> pose) and snapped to their wall / floor / ceiling.
5. Outputs in data/walkthrough/<name>/ :
     index.html              viewer: live 2D map (video) + 3D model with clickable findings
     live_map.mp4            video frame | 2D coverage map growing as the person walks
     coverage_map.png        final map: coverage, path, AI keyframes, numbered findings
     inspection_3d.glb       full clean SLAM model (no landmark cloud) + finding pins + keyframes
     inspection_3d_gaps.glb  open-top SLAM model + red/amber coverage-gap patches + pins
     walkthrough.json        stats, coverage over time, gaps to re-film, findings (3D)
     report.html / summary.md  inspection report of the keyframes
6. Property risk (condition grade, perils, repair cost, valuation impact, underwriting decision with the
   AI-inspected coverage as evidence completeness) -> walkthrough.json["risk"]; optional GCS / BigQuery / Pub/Sub.

Modes (--mode): both use the same AI models (Gemini scene + object names, LocateAnything, Gemma verifier,
Gemini second opinion), so the inspection results are the same; only the 3D build differs.
  advanced  full-detail 3D: 1.5 cm TSDF, 2 alignment passes, preview videos.
  lite      faster 3D (LITE_SLAM_ARGS): 3 cm TSDF, 1 alignment pass, no preview videos. Same frames,
            so camera tracking and finding positions are unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Lite only changes the 3D build; the AI models are the same in both modes.
MODE = "lite" if "--mode" in sys.argv[:-1] and sys.argv[sys.argv.index("--mode") + 1].lower() == "lite" else "advanced"
LITE_SLAM_ARGS = ["--voxel", "0.03", "--align-iters", "1", "--no-video"]

from backend.config.settings import get_settings  # noqa: E402
from backend.services import cloud_service, risk_service  # noqa: E402
from backend.services import report_service as reports  # noqa: E402
from backend.services.pipeline import InspectionPipeline  # noqa: E402
from backend.spatial import glb3d, map2d  # noqa: E402
from backend.spatial.coverage import compute_coverage, make_samples, mark_inspected, select_keyframes  # noqa: E402
from backend.spatial.localize import box_to_world, snap_to_surface  # noqa: E402
from backend.spatial.slam_data import load_slam_run  # noqa: E402

PROD = ROOT / "engine3d"
PROD_PY = Path(sys.executable)      # one environment for the app and the 3D engine
SLAM_OUT = Path(get_settings().slam_work_dir or Path(get_settings().data_dir) / "slam")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_slam_dir(video: Path) -> Path | None:
    stem = video.stem
    for c in (SLAM_OUT / stem, video.parent / f"{stem}_result_hq", video.parent / f"{stem}_result"):
        if (c / "05_trajectory" / "cameras.json").exists() and (c / "03_floor_plan" / "layout.json").exists():
            return c
    return None


def run_slam(video: Path, extra: list[str]) -> Path:
    s = get_settings()
    out = SLAM_OUT / video.stem
    log(f"SLAM: running the 3D engine on {video.name} (Depth Anything 3 {s.da3_model} + OpenCV SLAM) ...")
    cmd = [str(PROD_PY), "run_production.py", "--video", str(video.resolve()), "--output-dir", str(out),
           "--model", s.da3_model, "--fps", str(s.slam_fps), "--max-frames", str(s.slam_max_frames)] \
        + s.slam_extra_args.split() + extra
    r = subprocess.run(cmd, cwd=PROD)
    if r.returncode != 0 or not (out / "05_trajectory" / "cameras.json").exists():
        raise SystemExit(f"3D engine failed (exit {r.returncode}); see {out / 'run_log.txt'}")
    return out


def to_h264(frames_dir: Path, fps: float, out: Path) -> None:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", str(frames_dir / "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags", "+faststart", str(out)], check=True)


def transcode_h264(src: Path, dst: Path) -> bool:
    """The 3D engine writes MPEG-4 part 2 (not playable in browsers) -> H.264."""
    import imageio_ffmpeg
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-movflags", "+faststart", str(dst)])
    return r.returncode == 0 and dst.exists()


# (published name, 3D engine path, label) - copied next to the viewer so the web server can serve them
SLAM_MODELS = [("slam/room_slam.glb", "02_slam_model/room_slam_open_top.glb", "SLAM model (point layers + mesh)"),
               ("slam/room_clean.glb", "01_clean_model/room_clean.glb", "Clean room model (walls + furniture)"),
               ("slam/objects.glb", "01_clean_model/objects_only.glb", "Furniture / objects only")]
SLAM_MAPS = [("slam/floor_plan.svg", "03_floor_plan/floor_plan.svg", "2D floor plan"),
             ("slam/occupancy_map.png", "03_floor_plan/occupancy_map.png", "Occupancy map"),
             ("slam/navigation_map.png", "03_floor_plan/navigation_map.png", "Navigation map (A* path)"),
             ("slam/slam_model_top.jpg", "07_previews/slam_model_top.jpg", "SLAM model — top view"),
             ("slam/clean_model_top.jpg", "07_previews/clean_model_top.jpg", "Clean model — top view"),
             ("slam/structure_segmentation.jpg", "07_previews/structure_segmentation.jpg", "Structure segmentation"),
             ("slam/objects_gallery.jpg", "07_previews/objects_gallery.jpg", "Detected objects"),
             ("slam/model_vs_video.jpg", "07_previews/model_vs_video.jpg", "Model vs video"),
             ("slam/depth_examples.jpg", "06_depth/depth_examples.jpg", "Depth examples")]
SLAM_VIDEOS = [("slam/walkthrough_video_vs_model.mp4", "07_previews/walkthrough_video_vs_model.mp4", "Video vs 3D model"),
               ("slam/depth_preview.mp4", "06_depth/depth_preview.mp4", "Depth (Depth Anything 3)"),
               ("slam/slam_diagnostic.mp4", "07_previews/slam_diagnostic.mp4", "SLAM tracking diagnostic"),
               ("slam/orbit_clean_model.mp4", "07_previews/orbit_clean_model.mp4", "Orbit of the clean model")]


def collect_extras(out: Path, slam_dir: Path, video: Path, keyframes: list[dict], store) -> dict:
    """Copy everything worth showing into the walkthrough folder: SLAM models / maps / videos,
    the source video and the annotated AI keyframes. Returns the manifest for walkthrough.json."""
    (out / "slam").mkdir(exist_ok=True)
    (out / "annotated").mkdir(exist_ok=True)
    ex = {"models": [], "maps": [], "videos": [], "keyframes": []}
    for kind, items in (("models", SLAM_MODELS), ("maps", SLAM_MAPS)):
        for dst, src, label in items:
            if (slam_dir / src).exists():
                shutil.copy(slam_dir / src, out / dst)
                if dst == "slam/room_clean.glb":
                    glb3d.clean_glb_file(out / dst)
                if dst.endswith(".glb"):
                    glb3d.make_unlit(out / dst)
                ex[kind].append({"file": dst, "label": label})
    log("converting SLAM videos to H.264 for the browser ...")
    if video.exists():
        if transcode_h264(video, out / "source_video.mp4"):
            ex["videos"].append({"file": "source_video.mp4", "label": "Original walkthrough video"})
    for dst, src, label in SLAM_VIDEOS:
        if (slam_dir / src).exists() and transcode_h264(slam_dir / src, out / dst):
            ex["videos"].append({"file": dst, "label": label})
    for k in keyframes:
        d = store.dir(k["inspection_id"])
        ins = json.loads((d / "inspection.json").read_text(encoding="utf-8")) if (d / "inspection.json").exists() else {}
        ann = d / "annotated.jpg"
        item = {"inspection_id": k["inspection_id"], "time_s": k["time_s"], "image": k["image"],
                "status": ins.get("status", "unknown"), "room": ins.get("room", ""),
                "n_detected": sum(1 for x in ins.get("detections", []) if x.get("state") == "DETECTED"),
                "n_review": sum(1 for x in ins.get("detections", []) if x.get("state") != "DETECTED"),
                "error": ins.get("error") or "", "annotated": "", "report": f"/api/v1/inspection/{k['inspection_id']}/report"}
        if ann.exists():
            shutil.copy(ann, out / "annotated" / f"{k['inspection_id']}.jpg")
            item["annotated"] = f"annotated/{k['inspection_id']}.jpg"
        ex["keyframes"].append(item)
    return ex


def merge_repeats(findings: list[dict], radius: float = 0.6) -> list[dict]:
    """The same defect filmed from several keyframes is one finding: same component within `radius` metres
    (and the same defect, or either one generic "surface damage") -> keep the best, list the other frames."""
    rank = {"DETECTED": 0, "NEEDS_REVIEW": 1}
    kept: list[dict] = []
    for fd in sorted(findings, key=lambda x: (rank.get(x["state"], 2), x.get("defect_type") == "surface_damage")):
        p = None if fd.get("xyz") is None else np.asarray(fd["xyz"], float)
        dup = None
        if p is not None:
            for k in kept:
                if (k.get("xyz") is not None and k["component_key"] == fd["component_key"]
                        and (k["defect_type"] == fd["defect_type"]
                             or "surface_damage" in (k["defect_type"], fd["defect_type"]))
                        and np.linalg.norm(np.asarray(k["xyz"], float) - p) < radius):
                    dup = k
                    break
        if dup is None:
            kept.append(fd)
        else:
            dup.setdefault("also_seen", []).append({"issue_id": fd["issue_id"], "frame": fd["frame"],
                                                    "time_s": fd["time_s"], "crop": fd.get("crop", "")})
    kept.sort(key=lambda x: x["frame"])
    for i, fd in enumerate(kept, 1):
        fd["n"] = i
    if len(kept) < len(findings):
        log(f"merged {len(findings) - len(kept)} repeat sighting(s) of the same defect")
    return kept


def place_pin_tips(findings: list[dict], base: float = 0.35, min_gap: float = 0.28) -> None:
    """Pin head ("tip") of every finding: out of the surface along its normal, fanned out sideways and
    lengthened when another pin head is too close, so numbers in the 3D viewer never sit on each other."""
    placed: list[np.ndarray] = []
    for fd in findings:
        if fd.get("xyz") is None:
            fd["tip"] = None
            continue
        p = np.asarray(fd["xyz"], float)
        n = np.asarray(fd.get("normal") or [0, 1, 0], float)
        n = n / max(np.linalg.norm(n), 1e-6)
        side = np.cross(n, [0, 1, 0])
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-3 else np.array([1.0, 0, 0])
        tip = p + n * base
        for k in range(1, 9):
            if all(np.linalg.norm(tip - q) >= min_gap for q in placed):
                break
            sgn = 1 if k % 2 else -1
            tip = p + n * (base + 0.12 * ((k + 1) // 2)) + side * sgn * min_gap * ((k + 1) // 2)                 + np.array([0, 0.12 * ((k + 1) // 2), 0])
        placed.append(tip)
        fd["tip"] = [round(float(v), 3) for v in tip]
        fd["normal"] = [round(float(v), 4) for v in n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--slam-dir", default=None, help="existing 3D engine output folder")
    ap.add_argument("--run-slam", action="store_true", help="run the 3D engine if no SLAM output is found")
    ap.add_argument("--slam-args", default="", help="extra args for run_production.py, e.g. \"--fps 15 --max-depth 8\"")
    ap.add_argument("--name", default=None)
    ap.add_argument("--property", default=None)
    ap.add_argument("--mode", choices=["lite", "advanced"], default="advanced",
                    help="lite = faster, lighter 3D build; advanced = full-detail 3D (same AI models)")
    ap.add_argument("--max-keyframes", type=int, default=16)
    ap.add_argument("--good-range", type=float, default=3.0, help="max distance (m) that counts as inspectable")
    ap.add_argument("--no-video", action="store_true", help="skip live_map.mp4")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-make maps / video / GLBs / viewer from an existing walkthrough.json (no AI re-run)")
    a = ap.parse_args()

    video = Path(a.video).resolve()
    name = a.name or video.stem
    out = Path(get_settings().data_dir) / "walkthrough" / name
    prev = None
    if a.rebuild:
        prev = json.loads((out / "walkthrough.json").read_text(encoding="utf-8"))
        a.slam_dir = a.slam_dir or prev["slam_dir"]
    elif out.exists():
        shutil.rmtree(out)
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    t_all = time.perf_counter()

    # ---------------------------------------------------------------- 1. SLAM
    slam_dir = Path(a.slam_dir) if a.slam_dir else find_slam_dir(video)
    if slam_dir is None:
        if not a.run_slam:
            raise SystemExit(f"No 3D engine output for {video.name}. Add --run-slam or --slam-dir.")
        slam_dir = run_slam(video, (LITE_SLAM_ARGS if MODE == "lite" else []) + a.slam_args.split())
    log(f"SLAM run: {slam_dir}")
    run = load_slam_run(slam_dir)

    # ---------------------------------------------------------------- 2. coverage
    samples = make_samples(run.layout, 0.1)
    cov = compute_coverage(run, samples, good_range=a.good_range)
    log(f"coverage over {run.n} SLAM frames: {cov.stats()['overall']}")

    # ---------------------------------------------------------------- 3. keyframes + AI inspection
    cap = cv2.VideoCapture(str(video))
    frames = {}
    sharp = np.zeros(run.n)
    for i, vf in enumerate(run.video_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vf))
        ok, fr = cap.read()
        if not ok:
            continue
        if (fr.shape[1], fr.shape[0]) != run.image_size:
            fr = cv2.resize(fr, run.image_size, interpolation=cv2.INTER_AREA)
        frames[i] = fr
        sharp[i] = cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    cap.release()
    if prev:
        kfs = [k["frame"] for k in prev["keyframes"]]
    else:
        kfs = select_keyframes(cov, sharp, max_frames=a.max_keyframes)
    mark_inspected(cov, kfs)
    log(f"{len(kfs)} AI keyframes (greedy coverage): " + ", ".join(f"{run.times[f]:.1f}s" for f in kfs))

    pipe = InspectionPipeline()
    ids = []
    for f in ([] if prev else kfs):
        p = out / "keyframes" / f"{name}_t{run.times[f]:05.1f}s.jpg"
        cv2.imwrite(str(p), frames[f], [cv2.IMWRITE_JPEG_QUALITY, 97])
        ins = pipe.create(p.read_bytes(), p.name, property_id=a.property or name)
        ids.append(ins.inspection_id)
    inspections = {}
    if prev:
        ids = [k["inspection_id"] for k in prev["keyframes"]]
        inspections = {i: pipe.store.load(i) for i in ids}
    # One batch: each local model is loaded once for all keyframes (stage-major pipeline).
    log(f"AI inspection ({MODE}): scene={pipe.s.scene_backend} locate={pipe.s.locate_backend} "
        f"verify={pipe.s.verify_backend} second-opinion={pipe.s.validate_backend}")
    if ids and not prev:
        for ins in pipe.run_many(ids):
            inspections[ins.inspection_id] = ins
            log(f"  {ins.inspection_id} {ins.source_image}: {ins.status} room={ins.room} "
                f"detected={sum(d.state.value == 'DETECTED' for d in ins.detections)} "
                f"review={sum(d.state.value != 'DETECTED' for d in ins.detections)} {ins.error}")

    pipe.vision.release()          # stop the Gemma server / free LocateAnything before rendering
    pipe.localizer.release()

    # ---------------------------------------------------------------- 4. findings in 3D
    findings = list(prev["findings"]) if prev else []
    for f, iid in ([] if prev else zip(kfs, ids)):
        ins = inspections[iid]
        for d in ins.detections:
            b = d.localization
            xyz, dist = box_to_world(run, f, (b.x1, b.y1, b.x2, b.y2), run.image_size)
            surf, snapped, sd = snap_to_surface(xyz, samples) if xyz is not None else ("", None, None)
            normal = None
            if surf:
                i = int(np.argmin(np.linalg.norm(samples.xyz - snapped, axis=1)))
                normal = samples.normal[i].tolist()
            elif xyz is not None:   # on furniture / an object: point the pin back towards the camera
                v = run.c2w[f][:3, 3] - np.asarray(xyz, float)
                normal = (v / max(np.linalg.norm(v), 1e-6)).round(4).tolist()
            fx = float(run.K[f][0, 0]) if dist else 0.0
            size_m = [round((b.x2 - b.x1) * dist / fx, 2), round((b.y2 - b.y1) * dist / fx, 2)] if fx else None
            crop = pipe.store.dir(iid) / d.crop_path
            if crop.exists():
                shutil.copy(crop, out / "crops" / crop.name)
            findings.append({
                "n": len(findings) + 1, "issue_id": d.issue_id, "inspection_id": iid, "frame": int(f),
                "time_s": round(float(run.times[f]), 2), "component": d.component_label, "defect": d.defect.label,
                "component_key": d.component, "defect_type": d.defect.type,
                "state": d.state.value, "severity": d.defect.severity, "description": d.description,
                "review_reasons": d.review_reasons, "professional_review": d.professional_review,
                "xyz": None if xyz is None else [round(float(v), 3) for v in (snapped if surf else xyz)],
                "camera_distance_m": None if dist is None else round(dist, 2),
                "surface": surf, "surface_label": samples.groups.get(surf, {}).get("label", surf),
                "normal": normal, "crop": f"crops/{crop.name}" if crop.exists() else "", "size_m": size_m,
                "validation": d.validation, "models": d.models.model_dump()})
    findings = merge_repeats(findings)
    for fd in findings:   # older runs / objects: no surface normal -> point the pin towards the camera
        if fd.get("xyz") is not None and not fd.get("normal") and 0 <= fd.get("frame", -1) < run.n:
            v = run.c2w[fd["frame"]][:3, 3] - np.asarray(fd["xyz"], float)
            fd["normal"] = (v / max(np.linalg.norm(v), 1e-6)).round(4).tolist()
    place_pin_tips(findings)
    log(f"{len(findings)} findings placed in 3D")

    # ---------------------------------------------------------------- 5. outputs
    stats = cov.stats()
    gaps = cov.gaps()
    pm = map2d.PlanMap(run, cov)
    final = pm.render(None, kfs, findings)
    leg = map2d.legend(final.shape[1])
    lines = [f"Overall: seen {stats['overall']['seen_pct']}%, inspectable {stats['overall']['inspectable_pct']}%,"
             f" AI-inspected {stats['overall']['inspected_pct']}%", f"Findings: {len(findings)}  |  AI keyframes: {len(kfs)}"]
    lines += ["Not covered (re-film):"] + [f"  - {g['label']}: " + (f"{g['length_m']} m" if g.get('length_m') else
                                                                     f"{g['uncovered_m2']} m2") for g in gaps[:6]]
    panel = map2d.stats_panel(final.shape[1], 170 + 21 * len(lines), stats, float(run.times[-1]), lines)
    cv2.imwrite(str(out / "coverage_map.png"), np.vstack([final, leg, panel]))

    series = []
    for f in range(run.n):
        s = cov.stats(f)
        series.append({"t": round(float(run.times[f]), 2), "overall_seen": s["overall"]["seen_pct"],
                       "overall_inspectable": s["overall"]["inspectable_pct"],
                       **{f"{k}_seen": v["seen_pct"] for k, v in s["by_kind"].items()},
                       **{f"{k}_inspectable": v["inspectable_pct"] for k, v in s["by_kind"].items()}})

    if not a.no_video:
        log("rendering live map video ...")
        tmp = out / "_frames"
        tmp.mkdir()
        mh = final.shape[0]
        vw = 640
        kf_set = {f: iid for f, iid in zip(kfs, ids)}
        hold = None
        for f in range(run.n):
            m = pm.render(f, kfs, findings)
            fr = frames.get(f)
            if f in kf_set:
                ann = pipe.store.dir(kf_set[f]) / "annotated.jpg"
                if ann.exists():
                    hold = (cv2.imread(str(ann)), f, kf_set[f])
            vid = cv2.resize(fr if fr is not None else np.zeros((480, 848, 3), np.uint8), (vw, int(vw * 480 / 848)))
            if hold is not None and f - hold[1] <= 6:      # show the AI result for ~0.6 s after a keyframe
                vid = cv2.resize(hold[0], (vw, int(vw * hold[0].shape[0] / hold[0].shape[1])))
                map2d._put(vid, f"AI inspection {hold[2]}", (10, 26), 0.7, (180, 0, 180), 2)
            s = cov.stats(f)
            nf = sum(1 for x in findings if x["frame"] <= f)
            left_h = max(mh, vid.shape[0] + 220)
            panel = map2d.stats_panel(vw, left_h - vid.shape[0], s, float(run.times[f]),
                                      [f"Overall seen {s['overall']['seen_pct']}%  inspectable {s['overall']['inspectable_pct']}%",
                                       f"Findings so far: {nf}", f"Pose: {run.pose_state[f]}  frame quality: {run.frame_quality[f]}"])
            left = np.vstack([vid, panel])
            right = np.vstack([m, np.full((left_h - m.shape[0], m.shape[1], 3), 255, np.uint8)]) if left_h > m.shape[0] else m
            canvas = np.hstack([left, right])
            cv2.imwrite(str(tmp / f"{f:05d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        slam_fps = run.n / max(1e-6, float(run.times[-1] - run.times[0])) if run.n > 1 else 10
        to_h264(tmp, slam_fps, out / "live_map.mp4")
        shutil.rmtree(tmp)

    log("building 3D GLBs ...")
    # Default = clean SLAM model like V3's clean_room_furnished_slam.glb: textured walls / floor / ceiling, named
    # furniture, trajectory, cameras, finding pins - no landmark cloud, no raw fill, no unlabelled fragments.
    glb3d.build_glb(run, cov, findings, kfs, out / "inspection_3d.glb", open_top=False, with_gaps=False,
                    drop=("slam_landmarks",), clean=True)
    glb3d.build_glb(run, cov, findings, kfs, out / "inspection_3d_gaps.glb", open_top=True, with_gaps=True)
    # "Walk inside": the same clean room (with ceiling) + finding pins only - no SLAM overlays in front of the camera.
    clean_glb = Path(slam_dir) / "01_clean_model" / "room_clean.glb"
    if clean_glb.exists():
        glb3d.build_glb(run, cov, findings, kfs, out / "inspection_3d_inside.glb", with_gaps=False,
                        base=clean_glb, with_keyframes=False, clean=True)
        info = glb3d.clean_presentation(trimesh.load(str(clean_glb), force="scene"))
        n_frag = sum(n.startswith("object") for n in info["removed"])
        log(f"clean model: removed raw fill + {n_frag} unlabelled fragments (fill share {info['context_share']})"
            if info["applied"] else
            f"clean model: kept raw fill (room shell unreliable, fill share {info['context_share']})")
    (out / "inspection_3d_clean.glb").unlink(missing_ok=True)   # old layout

    done = [inspections[i] for i in ids if inspections[i].status == "done"]
    if done:
        (out / "report.html").write_text(reports.render_html(done, [pipe.store.dir(i.inspection_id) for i in done],
                                                             title=f"Walkthrough Inspection — {name}",
                                                             property_id=a.property or name), encoding="utf-8")
    warnings = []
    if run.ceiling > 3.3:
        warnings.append(f"Reconstructed ceiling height is {run.ceiling:.2f} m. If the real ceiling is lower, the SLAM "
                        f"metric scale is too large (it assumes the camera was held {run.extras['config'].get('camera_height', 1.5)} m "
                        "above the floor); distances are inflated and 'inspectable' (<= 3 m) is under-counted. "
                        "Re-run SLAM with --slam-args \"--camera-height <real height>\".")
    if stats["overall"]["seen_pct"] < 50:
        warnings.append("Less than half of the mapped surfaces were seen - the walkthrough did not cover the whole area.")
    W = {"name": name, "mode": MODE, "built": time.strftime("%Y%m%d%H%M%S"), "warnings": warnings, "video": str(video), "slam_dir": str(slam_dir), "duration_s": round(float(run.times[-1]), 2),
         "slam_frames": run.n, "image_size": run.image_size, "ceiling_m": run.ceiling,
         "rooms": [{"id": r["id"], "name": r["name"], "area_m2": r["area_m2"], "size_m": r["size_m"]}
                   for r in run.layout.get("rooms", [])],
         "coverage": stats, "coverage_params": cov.params, "gaps": gaps, "series": series,
         "keyframes": [{"frame": int(f), "time_s": round(float(run.times[f]), 2), "inspection_id": i,
                        "image": f"keyframes/{name}_t{run.times[f]:05.1f}s.jpg",
                        "position": [round(float(v), 3) for v in run.c2w[f][:3, 3]]} for f, i in zip(kfs, ids)],
         "findings": findings, "runtime_s": round(time.perf_counter() - t_all, 1),
         "files": {"live_map": "live_map.mp4" if not a.no_video else None, "coverage_map": "coverage_map.png",
                   "glb": "inspection_3d.glb", "glb_gaps": "inspection_3d_gaps.glb",
                   "glb_inside": "inspection_3d_inside.glb" if (out / "inspection_3d_inside.glb").exists() else None, "report": "report.html"}}
    # Camera of every SLAM frame (OpenCV convention: +z forward) -> the viewer's "walk inside" mode puts the
    # browser camera exactly where the video camera was, like Production's "clean 3D model, same camera".
    fy = float(np.median(run.K[:, 1, 1]))
    W["camera"] = {"fov_v_deg": round(float(np.degrees(2 * np.arctan(run.image_size[1] / 2 / fy))), 2),
                   "path": [{"t": round(float(run.times[f]), 3),
                             "p": [round(float(v), 3) for v in run.c2w[f][:3, 3]],
                             "f": [round(float(v), 4) for v in run.c2w[f][:3, 2]]} for f in range(run.n)]}
    W["extras"] = collect_extras(out, Path(slam_dir), video, W["keyframes"], pipe.store)
    # Property-level risk: all findings of the walkthrough; AI-inspected coverage = evidence completeness.
    adapter = [{"issue_id": x["issue_id"], "state": x["state"], "component": x.get("component_key", ""),
                "defect": {"type": x.get("defect_type", "default"), "severity": x.get("severity")}} for x in findings]
    # Evidence-integrity: one summary flag per kind (not one line per keyframe)
    by_code: dict[str, list[tuple[str, dict]]] = {}
    for i in ids:
        for fl in ((inspections[i].integrity or {}).get("flags", []) if i in inspections else []):
            if fl.get("level") == "high":
                by_code.setdefault(fl["code"], []).append((i, fl))
    flags = []
    for code, items in by_code.items():
        if code == "duplicate_other_property":
            props = sorted({fl["message"].split("for property ", 1)[-1].split(" (")[0] for _, fl in items})
            msg = (f"{len(items)} of {len(ids)} keyframes match photos already submitted for "
                   f"property {', '.join(props)} - same video reused for another property?")
        else:
            msg = f"{items[0][1]['message']} ({len(items)} of {len(ids)} keyframes)"
        flags.append({"code": code, "level": "high", "message": msg, "keyframes": [i for i, _ in items]})
    W["property_id"] = a.property or name
    W["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    W["risk"] = risk_service.assess_findings(adapter, get_settings(), integrity_flags=flags,
                                             coverage_pct=stats["overall"]["inspected_pct"],
                                             n_images=max(1, len(ids)))
    # Gemini documentation: property-level underwriter summary from the walkthrough record
    try:
        from backend.api.assistant import SUMMARY_TASK, SYSTEM
        rec = {k: W.get(k) for k in ("name", "property_id", "duration_s", "rooms", "risk", "gaps")}
        rec["coverage"] = stats["overall"]
        rec["findings"] = [{k: x.get(k) for k in ("issue_id", "component", "defect", "severity", "state",
                                                    "surface_label", "description", "validation")} for x in findings]
        W["ai_summary"] = pipe.vision.text(
            SYSTEM.format(currency=get_settings().currency) + "\n\nVIDEO WALKTHROUGH RECORD (3D map + coverage + "
            "findings):\n" + json.dumps(rec, default=str)[:60000] + "\n\nTASK:\n" + SUMMARY_TASK +
            " Also state which surfaces must be re-filmed.", max_tokens=900).strip()
        log("Gemini property summary written")
    except Exception as e:  # noqa: BLE001
        W["ai_summary"] = ""
        log(f"Gemini summary skipped: {e}")
    if done:   # re-render the report with the property-level risk block
        (out / "report.html").write_text(reports.render_html(done, [pipe.store.dir(i.inspection_id) for i in done],
                                                             title=f"Walkthrough Inspection — {name}",
                                                             property_id=a.property or name, property_risk=W["risk"]),
                                         encoding="utf-8")
    (out / "walkthrough.json").write_text(json.dumps(W, indent=2, default=float), encoding="utf-8")
    cloud_service.publish_walkthrough(name, out, {"property_id": W["property_id"], "risk": W["risk"],
                                                  "n_findings": len(findings),
                                                  "n_detected": sum(1 for x in findings if x["state"] == "DETECTED"),
                                                  "coverage_pct": stats["overall"]["inspected_pct"]}, get_settings())
    shutil.copy(ROOT / "frontend" / "walkthrough.html", out / "index.html")
    md = [f"# Walkthrough inspection — {name}", "",
          f"- Video: `{video}` ({W['duration_s']} s, {run.n} SLAM frames); SLAM: `{slam_dir}`",
          f"- Rooms: " + ", ".join(f"{r['name']} {r['area_m2']} m2" for r in W["rooms"]),
          f"- Coverage: seen {stats['overall']['seen_pct']}%, inspectable {stats['overall']['inspectable_pct']}%, "
          f"inspected by AI {stats['overall']['inspected_pct']}%", ""] + [f"> **Warning:** {w}" for w in warnings] + [""]
    md += ["| Surface | Seen | Inspectable | AI-inspected |", "|---|---:|---:|---:|"]
    md += [f"| {v['label']} | {v['seen_pct']}% | {v['inspectable_pct']}% | {v['inspected_pct']}% |"
           for v in stats["by_group"].values()]
    md += ["", "## Not covered — film again", ""] + [
        f"- {g['label']}: " + (f"{g['length_m']} m (t {g['from_t']}..{g['to_t']} m)" if g.get("length_m") else
                               f"{g['uncovered_m2']} m2 not inspectable") for g in gaps] + ["", "## Findings", ""]
    md += [f"- #{x['n']} {x['component']} — {x['defect']} ({x['state']}) at {x['time_s']} s on {x['surface_label'] or '?'}: "
           f"{x['description']}" for x in findings] or ["- none"]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    log(f"done in {time.perf_counter() - t_all:.0f}s -> {out}")
    log(f"view: start the app, then open /walkthroughs/{name}/")
    if get_settings().gcs_bucket or get_settings().bq_dataset or get_settings().pubsub_topic:
        time.sleep(20)   # give the background GCS / BigQuery / Pub/Sub upload time before the process exits


if __name__ == "__main__":
    main()
