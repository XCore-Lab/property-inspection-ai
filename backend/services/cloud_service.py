"""Google Cloud integrations. Each one is OFF until its env var is set, and a failure is logged,
never raised, so the inspection result is always delivered.

  PI_GCS_BUCKET   Cloud Storage: archive every inspection / walkthrough folder (gs://<bucket>/<kind>/<id>/...)
  PI_BQ_DATASET   BigQuery: one row per inspection / walkthrough in <project>.<dataset>.inspections
                  (risk analytics: portfolio condition, perils, repair exposure, fraud flags)
  PI_PUBSUB_TOPIC Pub/Sub: event {"type": "inspection.completed", ...} for downstream underwriting /
                  claims systems (event-driven processing)
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.config.settings import Settings

log = logging.getLogger(__name__)
_clients: dict[str, object] = {}
_lock = threading.Lock()

BQ_SCHEMA = [
    ("record_id", "STRING"), ("kind", "STRING"), ("property_id", "STRING"), ("created_at", "TIMESTAMP"),
    ("room", "STRING"), ("n_findings", "INTEGER"), ("n_detected", "INTEGER"), ("condition_score", "FLOAT"),
    ("grade", "STRING"), ("risk_level", "STRING"), ("decision", "STRING"), ("repair_cost_low", "FLOAT"),
    ("repair_cost_high", "FLOAT"), ("currency", "STRING"), ("valuation_impact_low_pct", "FLOAT"),
    ("valuation_impact_high_pct", "FLOAT"), ("premium_loading_pct", "FLOAT"), ("collateral_risk", "STRING"),
    ("integrity_level", "STRING"), ("integrity_flags", "STRING"), ("perils", "STRING"),
    ("coverage_pct", "FLOAT"), ("model", "STRING"), ("report_uri", "STRING"),
]


def _client(kind: str, s: Settings):
    with _lock:
        if kind not in _clients:
            project = s.gcp_project or None
            if kind == "gcs":
                from google.cloud import storage
                _clients[kind] = storage.Client(project=project)
            elif kind == "bq":
                from google.cloud import bigquery
                _clients[kind] = bigquery.Client(project=project)
            elif kind == "pubsub":
                from google.cloud import pubsub_v1
                _clients[kind] = pubsub_v1.PublisherClient()
        return _clients[kind]


def status(s: Settings) -> dict:
    return {"gcs_bucket": s.gcs_bucket or None, "bigquery_dataset": s.bq_dataset or None,
            "pubsub_topic": s.pubsub_topic or None, "project": s.gcp_project or None,
            "gemini": "vertex" if s.gemini_vertex else "api_key"}


def upload_dir(local: Path, prefix: str, s: Settings) -> str:
    if not s.gcs_bucket:
        return ""
    try:
        bucket = _client("gcs", s).bucket(s.gcs_bucket)
        for f in local.rglob("*"):
            if f.is_file() and f.stat().st_size < 200 * 2**20:
                bucket.blob(f"{prefix}/{f.relative_to(local).as_posix()}").upload_from_filename(str(f))
        return f"gs://{s.gcs_bucket}/{prefix}"
    except Exception as e:  # noqa: BLE001
        log.warning("GCS upload failed for %s: %s", prefix, e)
        return ""


def bq_insert(row: dict, s: Settings) -> None:
    if not s.bq_dataset:
        return
    try:
        from google.cloud import bigquery
        bq = _client("bq", s)
        table_id = f"{bq.project}.{s.bq_dataset}.inspections"
        try:
            bq.get_table(table_id)
        except Exception:  # noqa: BLE001  (first run: create dataset + table)
            bq.create_dataset(s.bq_dataset, exists_ok=True)
            bq.create_table(bigquery.Table(table_id, schema=[bigquery.SchemaField(n, t) for n, t in BQ_SCHEMA]),
                            exists_ok=True)
        errors = bq.insert_rows_json(table_id, [row])
        if errors:
            log.warning("BigQuery insert errors: %s", errors)
    except Exception as e:  # noqa: BLE001
        log.warning("BigQuery insert failed: %s", e)


def publish(event: dict, s: Settings) -> None:
    if not s.pubsub_topic:
        return
    try:
        pub = _client("pubsub", s)
        topic = s.pubsub_topic if s.pubsub_topic.startswith("projects/") else \
            pub.topic_path(s.gcp_project, s.pubsub_topic)
        pub.publish(topic, json.dumps(event, default=str).encode("utf-8"), type=event.get("type", "")).result(30)
    except Exception as e:  # noqa: BLE001
        log.warning("Pub/Sub publish failed: %s", e)


def risk_row(record_id: str, kind: str, property_id: str, risk: dict | None, *, room: str = "",
             n_findings: int = 0, n_detected: int = 0, integrity: dict | None = None,
             coverage_pct: float | None = None, model: str = "", report_uri: str = "") -> dict:
    r = risk or {}
    v = (r.get("valuation") or {}).get("impact_pct") or [0, 0]
    return {
        "record_id": record_id, "kind": kind, "property_id": property_id or "",
        "created_at": datetime.now(timezone.utc).isoformat(), "room": room,
        "n_findings": n_findings, "n_detected": n_detected,
        "condition_score": r.get("condition_score"), "grade": r.get("grade"), "risk_level": r.get("risk_level"),
        "decision": r.get("decision"), "repair_cost_low": (r.get("repair_cost") or {}).get("low"),
        "repair_cost_high": (r.get("repair_cost") or {}).get("high"),
        "currency": (r.get("repair_cost") or {}).get("currency"),
        "valuation_impact_low_pct": v[0], "valuation_impact_high_pct": v[1],
        "premium_loading_pct": (r.get("insurance") or {}).get("premium_loading_pct"),
        "collateral_risk": (r.get("mortgage") or {}).get("collateral_risk"),
        "integrity_level": (integrity or {}).get("level", ""),
        "integrity_flags": json.dumps([f.get("code") for f in (integrity or {}).get("flags", [])]),
        "perils": json.dumps([p.get("category") for p in (r.get("insurance") or {}).get("perils", [])]),
        "coverage_pct": coverage_pct, "model": model, "report_uri": report_uri,
    }


def _run_async(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


def publish_inspection(ins, folder: Path, s: Settings) -> None:
    if not (s.gcs_bucket or s.bq_dataset or s.pubsub_topic):
        return

    def work():
        uri = upload_dir(folder, f"inspections/{ins.inspection_id}", s)
        row = risk_row(ins.inspection_id, "photo", ins.property_id, ins.risk, room=ins.room,
                       n_findings=len(ins.detections),
                       n_detected=sum(1 for d in ins.detections if d.state.value == "DETECTED"),
                       integrity=ins.integrity, model=(ins.options.get("models") or {}).get("verification", ""),
                       report_uri=uri)
        bq_insert(row, s)
        publish({"type": "inspection.completed", **row}, s)
    _run_async(work)


def publish_walkthrough(name: str, folder: Path, summary: dict, s: Settings) -> None:
    if not (s.gcs_bucket or s.bq_dataset or s.pubsub_topic):
        return

    def work():
        uri = upload_dir(folder, f"walkthroughs/{name}", s)
        row = risk_row(name, "walkthrough", summary.get("property_id", ""), summary.get("risk"),
                       n_findings=summary.get("n_findings", 0), n_detected=summary.get("n_detected", 0),
                       integrity=None, coverage_pct=summary.get("coverage_pct"), model=s.gemini_model,
                       report_uri=uri)
        bq_insert(row, s)
        publish({"type": "walkthrough.completed", **row}, s)
    _run_async(work)
