"""3D inspection model: the 3D engine SLAM GLB (clean room + trajectories + camera
poses + landmarks) plus inspection layers:

  inspection_gaps      coloured patches 1.5 cm in front of every surface cell that was NOT
                       inspectable: red = never seen, amber = seen only far away / obliquely
  inspection_findings  a pointer (arrow) at each finding, its tip touching the defect (red = detected,
                       orange = needs review), head sphere at the tip
  inspection_keyframes magenta frustums at the frames the AI inspected
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np
import trimesh

from backend.spatial.coverage import SURF_CEIL, Coverage
from backend.spatial.slam_data import SlamRun

RGBA = {"unseen": (225, 35, 35, 255), "seen": (255, 165, 0, 255), "DETECTED": (220, 38, 38, 255),
        "NEEDS_REVIEW": (234, 150, 12, 255), "UNKNOWN": (37, 99, 235, 255), "kf": (200, 0, 200, 255)}


def _colored(m: trimesh.Trimesh, rgba) -> trimesh.Trimesh:
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.tile(np.array(rgba, np.uint8), (len(m.vertices), 1)))
    return m


def gap_patches(cov: Coverage, include_ceiling: bool) -> trimesh.Trimesh | None:
    S = cov.samples
    st = cov.state()
    keep = st < 2
    if not include_ceiling:
        keep &= S.kind != SURF_CEIL
    idx = np.where(keep)[0]
    if not len(idx):
        return None
    h = S.step * 0.47
    V, F, Cc = [], [], []
    for n, i in enumerate(idx):
        p, nrm = S.xyz[i] + S.normal[i] * 0.015, S.normal[i]
        a = np.array([0, 1, 0.0]) if abs(nrm[1]) < 0.9 else np.array([1, 0, 0.0])
        u = np.cross(nrm, a); u /= np.linalg.norm(u)
        v = np.cross(nrm, u)
        V += [p - u * h - v * h, p + u * h - v * h, p + u * h + v * h, p - u * h + v * h]
        b = 4 * n
        # one-sided, facing into the room: from outside the open-top model you look through them
        tri = [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        if np.dot(np.cross(V[b + 1] - V[b], V[b + 2] - V[b]), nrm) < 0:
            tri = [[t[0], t[2], t[1]] for t in tri]
        F += tri
        Cc += [RGBA["unseen"] if st[i] == 0 else RGBA["seen"]] * 4
    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.array(Cc, np.uint8))
    return m


def finding_pins(findings: list[dict]) -> trimesh.Trimesh | None:
    parts = []
    for fd in findings:
        if fd.get("xyz") is None:
            continue
        p = np.asarray(fd["xyz"], float)
        nrm = np.asarray(fd.get("normal") or [0, 1, 0], float)
        nrm = nrm / max(np.linalg.norm(nrm), 1e-6)
        tip = np.asarray(fd["tip"], float) if fd.get("tip") else p + nrm * 0.35
        col = RGBA.get(fd["state"], RGBA["UNKNOWN"])
        # pointer: head at the tip, shaft, and an arrowhead whose point touches the defect
        d = p - tip
        L = float(np.linalg.norm(d))
        if L < 1e-3:
            continue
        u = d / L
        cone_h = min(0.12, 0.4 * L)
        base = p - u * cone_h
        parts.append(_colored(trimesh.creation.cylinder(radius=0.012, segment=[tip, base], sections=10), col))
        cone = trimesh.creation.cone(radius=0.035, height=cone_h, sections=16)   # apex at +z
        cone.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], u))
        parts.append(_colored(cone.apply_translation(base), col))
        parts.append(_colored(trimesh.creation.icosphere(2, radius=0.06).apply_translation(tip), col))
    return trimesh.util.concatenate(parts) if parts else None


def keyframe_frustums(run: SlamRun, frames: list[int], s: float = 0.14) -> trimesh.Trimesh | None:
    parts = []
    for f in frames:
        c2w = run.c2w[f]
        R, t = c2w[:3, :3], c2w[:3, 3]
        c = np.array([[-s, -.6 * s, 1.2 * s], [s, -.6 * s, 1.2 * s], [s, .6 * s, 1.2 * s], [-s, .6 * s, 1.2 * s]])
        pts = [t] + [R @ q + t for q in c]
        for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
            parts.append(trimesh.creation.cylinder(radius=s * 0.06, segment=[pts[a], pts[b]], sections=6))
    return _colored(trimesh.util.concatenate(parts), RGBA["kf"]) if parts else None


UNLABELLED = re.compile(r"^object_\d+_object$")
MAX_CONTEXT_SHARE = 0.30


def clean_presentation(scene: trimesh.Scene) -> dict:
    """V3-style clean room (like clean_room_furnished_slam.glb): straight textured walls / floor / ceiling + named
    furniture only. Removes Production's raw 'context_measured_surfaces' fill and unlabelled object fragments
    (ceiling beams, floor speckles, shards seen through doorways). Only done when the room shell is reliable:
    if the raw fill is a large share of the surfaces (failed / partial layout) everything is kept."""
    geo = {n: scene.geometry[scene.graph[n][1]] for n in scene.graph.nodes_geometry}
    shell = sum(m.area for n, m in geo.items() if n.startswith("structure_"))
    ctx = geo.get("context_measured_surfaces")
    share = (ctx.area / shell) if (ctx is not None and shell > 0) else 0.0
    info = {"context_share": round(float(share), 3), "applied": bool(shell > 0 and share <= MAX_CONTEXT_SHARE), "removed": []}
    if info["applied"]:
        for n in list(geo):
            if n == "context_measured_surfaces" or UNLABELLED.match(n):
                scene.delete_geometry(scene.graph[n][1])
                info["removed"].append(n)
    return info


def clean_glb_file(path: Path) -> dict:
    scene = trimesh.load(str(path), force="scene")
    info = clean_presentation(scene)
    if info["applied"]:
        scene.export(str(path))
    return info


def make_unlit(path: Path) -> Path:
    """Mark every material KHR_materials_unlit: the textures are photographs (lighting already baked in), so the
    browser must show them as filmed - like Production's 'clean 3D model, same camera' renders - not re-lit."""
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    for m in j.get("materials", []):
        m.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    used = j.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    rest = b[20 + n:]
    out = b[:12] + struct.pack("<I", len(js)) + b"JSON" + js + rest
    out = out[:8] + struct.pack("<I", len(out)) + out[12:]
    Path(path).write_bytes(out)
    return Path(path)


def build_glb(run: SlamRun, cov: Coverage, findings: list[dict], keyframes: list[int], out_path: Path,
              open_top: bool = True, with_gaps: bool = True, drop: tuple[str, ...] = (),
              base: Path | None = None, with_keyframes: bool = True, clean: bool = False) -> Path:
    base = base or run.glb(open_top)
    scene = trimesh.load(str(base), force="scene")
    for node in [n for n in scene.graph.nodes_geometry if any(d in n for d in drop)]:
        scene.delete_geometry(scene.graph[node][1])
    if clean:
        clean_presentation(scene)
    layers = {"inspection_findings": finding_pins(findings)}
    if with_keyframes:
        layers["inspection_keyframes"] = keyframe_frustums(run, keyframes)
    if with_gaps:
        layers["inspection_gaps"] = gap_patches(cov, include_ceiling=not open_top)
    for name, m in layers.items():
        if m is not None:
            scene.add_geometry(m, node_name=name, geom_name=name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))
    return make_unlit(out_path)
