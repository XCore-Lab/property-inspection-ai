"""Export the layout floor raster from
09_debug/work/model.pkl to a plain npz (the pickle needs Production's classes)."""
import pickle
import sys
from pathlib import Path

import numpy as np

slam_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # engine3d/ (v1prod package)
m = pickle.load(open(slam_dir / "09_debug" / "work" / "model.pkl", "rb"))
lay = m["lay"]
g = lay["grid"]
np.savez_compressed(out, floor=lay["floor"].astype(bool), floor_observed=lay["floor_observed"].astype(bool),
                    lo=np.asarray(g.lo, float), cell=float(g.cell), ceiling_h=float(lay["ceiling_h"]))
print("ok", lay["floor"].shape, g.lo, g.cell)
