"""Single GPU worker thread: inspections are queued and processed one at a time
(the 6 GB GPU cannot run two inspections concurrently). Queued jobs are
drained stage-major in small batches to minimise model swaps."""
from __future__ import annotations

import logging
import queue
import threading

from backend.services.pipeline import InspectionPipeline

log = logging.getLogger(__name__)


class JobRunner:
    def __init__(self, pipeline: InspectionPipeline, max_batch: int = 4):
        self.pipeline = pipeline
        self.q: queue.Queue[str] = queue.Queue()
        self.max_batch = max_batch
        self.current: list[str] = []
        self._t = threading.Thread(target=self._loop, daemon=True, name="inspection-worker")
        self._t.start()

    def submit(self, inspection_id: str) -> int:
        self.q.put(inspection_id)
        return self.q.qsize()

    def _loop(self) -> None:
        while True:
            ids = [self.q.get()]
            while len(ids) < self.max_batch:
                try:
                    ids.append(self.q.get_nowait())
                except queue.Empty:
                    break
            self.current = ids
            try:
                self.pipeline.run_many(ids)
            except Exception:  # noqa: BLE001  (run_many already marks failures)
                log.exception("batch failed: %s", ids)
            finally:
                self.current = []
