"""Save / load the stage-1 state (open3d geometry as PLY, everything else pickled)
so the later stages can be re-run without repeating DA3 / SLAM / TSDF."""

import os
import pickle

import numpy as np
import open3d as o3d

_MESHES = ("model", "raw_model")
_CLOUDS = ("pcd",)


def save(state, work):
    os.makedirs(work, exist_ok=True)
    rest = {}
    for k, v in state.items():
        if k in _MESHES:
            o3d.io.write_triangle_mesh(f"{work}/state_{k}.ply", v)
        elif k in _CLOUDS:
            o3d.io.write_point_cloud(f"{work}/state_{k}.ply", v)
        elif k in ("depth_up",):
            np.save(f"{work}/state_{k}.npy", v.astype(np.float16))
        else:
            rest[k] = v
    with open(f"{work}/state.pkl", "wb") as f:
        pickle.dump(rest, f, protocol=4)


def load(work):
    with open(f"{work}/state.pkl", "rb") as f:
        state = pickle.load(f)
    for k in _MESHES:
        m = o3d.io.read_triangle_mesh(f"{work}/state_{k}.ply")
        m.compute_vertex_normals()
        state[k] = m
    for k in _CLOUDS:
        state[k] = o3d.io.read_point_cloud(f"{work}/state_{k}.ply")
    state["depth_up"] = np.load(f"{work}/state_depth_up.npy").astype(np.float32)
    return state


def exists(work):
    return os.path.exists(f"{work}/state.pkl")
