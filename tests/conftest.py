import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config.settings import Settings, set_settings  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    s = Settings(data_dir=str(tmp_path / "data"), mock_models=True, locate_backend="mock", vlm_backend="mock",
                 scene_backend="mock", verify_backend="mock", validate_backend="mock", photo_upload=True)
    set_settings(s)
    return s


@pytest.fixture()
def tax(settings):
    from backend.services.taxonomy import get_taxonomy
    return get_taxonomy(settings.taxonomy_path)


def make_image(w=800, h=600, crack=True) -> bytes:
    rng = np.random.default_rng(0)
    a = (rng.normal(200, 6, (h, w, 3))).clip(0, 255).astype(np.uint8)
    if crack:
        import cv2
        pts = np.array([[w * 0.3, h * 0.2], [w * 0.35, h * 0.4], [w * 0.32, h * 0.6], [w * 0.4, h * 0.8]], np.int32)
        cv2.polylines(a, [pts], False, (40, 40, 40), 2)
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG")
    return b.getvalue()


@pytest.fixture()
def image_bytes():
    return make_image()
