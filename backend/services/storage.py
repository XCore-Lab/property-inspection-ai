"""On-disk inspection store: data/outputs/<inspection_id>/{inspection.json, original.*, ...}."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from backend.models.schemas import Inspection

_lock = threading.Lock()
ID_RE = re.compile(r"^INS-\d{4,}$")


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        with _lock:
            ctr = self.root / ".counter"
            n = int(ctr.read_text()) if ctr.exists() else 0
            existing = [int(p.name[4:]) for p in self.root.glob("INS-*") if ID_RE.match(p.name)]
            n = max([n] + existing) + 1
            ctr.write_text(str(n))
            d = self.root / f"INS-{n:04d}"
            d.mkdir(parents=True, exist_ok=True)
            return d.name

    def dir(self, inspection_id: str) -> Path:
        if not ID_RE.match(inspection_id):
            raise KeyError(inspection_id)
        return self.root / inspection_id

    def save(self, ins: Inspection) -> None:
        d = self.dir(ins.inspection_id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "inspection.json.tmp"
        tmp.write_text(ins.model_dump_json(indent=2), encoding="utf-8")
        # Windows: replace fails while a reader (API poll) has the target open - retry briefly.
        for attempt in range(50):
            try:
                tmp.replace(d / "inspection.json")
                return
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.02)

    def load(self, inspection_id: str) -> Inspection:
        p = self.dir(inspection_id) / "inspection.json"
        if not p.exists():
            raise KeyError(inspection_id)
        for attempt in range(50):
            try:
                return Inspection.model_validate_json(p.read_text(encoding="utf-8"))
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.02)
        raise KeyError(inspection_id)  # unreachable

    def list(self, property_id: str | None = None) -> list[Inspection]:
        out = []
        for p in sorted(self.root.glob("INS-*/inspection.json")):
            try:
                ins = Inspection.model_validate_json(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if property_id is None or ins.property_id == property_id:
                out.append(ins)
        return out

    def write_raw(self, inspection_id: str, name: str, content: str | dict | list) -> None:
        d = self.dir(inspection_id) / "raw"
        d.mkdir(exist_ok=True)
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, default=str)
        (d / name).write_text(content, encoding="utf-8")
