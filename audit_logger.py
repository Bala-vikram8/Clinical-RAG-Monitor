import sqlite3
import json
from datetime import datetime
from typing import List, Optional
from monitor.models import MonitoringResult, ClinicalAlert


DB_PATH = "healthcare_rag_monitor.db"


class AuditLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_results (
                    result_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    query_id TEXT,
                    response_id TEXT,
                    retrieval_quality_score REAL,
                    faithfulness_score REAL,
                    guideline_freshness_score REAL,
                    overall_safety_score REAL,
                    retrieval_passed INTEGER,
                    faithfulness_passed INTEGER,
                    freshness_passed INTEGER,
                    overall_passed INTEGER,
                    recommendations TEXT,
                    audit_trail TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clinical_alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    query_id TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    details TEXT,
                    action_required TEXT,
                    escalate_to_clinician INTEGER,
                    resolved INTEGER DEFAULT 0,
                    resolved_by TEXT,
                    resolved_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_health_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    total_docs INTEGER,
                    stale_docs INTEGER,
                    avg_freshness_score REAL,
                    critical_stale INTEGER,
                    snapshot_data TEXT
                )
            """)

    def log_result(self, result: MonitoringResult):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO monitoring_results
                (result_id, timestamp, query_id, response_id,
                 retrieval_quality_score, faithfulness_score,
                 guideline_freshness_score, overall_safety_score,
                 retrieval_passed, faithfulness_passed, freshness_passed,
                 overall_passed, recommendations, audit_trail)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result.result_id, result.timestamp, result.query_id,
                    result.response_id, result.retrieval_quality_score,
                    result.faithfulness_score, result.guideline_freshness_score,
                    result.overall_safety_score,
                    int(result.retrieval_passed), int(result.faithfulness_passed),
                    int(result.freshness_passed), int(result.overall_passed),
                    json.dumps(result.recommendations),
                    json.dumps(result.audit_trail),
                ),
            )
            for alert in result.alerts:
                conn.execute(
                    """INSERT OR REPLACE INTO clinical_alerts
                    (alert_id, timestamp, query_id, alert_type, severity,
                     message, details, action_required, escalate_to_clinician)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        alert.alert_id, alert.timestamp, alert.query_id,
                        alert.alert_type.value, alert.severity.value,
                        alert.message, json.dumps(alert.details),
                        alert.action_required, int(alert.escalate_to_clinician),
                    ),
                )

    def log_alert(self, alert: ClinicalAlert):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO clinical_alerts
                (alert_id, timestamp, query_id, alert_type, severity,
                 message, details, action_required, escalate_to_clinician)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    alert.alert_id, alert.timestamp, alert.query_id,
                    alert.alert_type.value, alert.severity.value,
                    alert.message, json.dumps(alert.details),
                    alert.action_required, int(alert.escalate_to_clinician),
                ),
            )

    def log_kb_snapshot(self, total_docs: int, stale_docs: int, avg_freshness: float, critical_stale: bool, data: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO kb_health_snapshots
                (timestamp, total_docs, stale_docs, avg_freshness_score, critical_stale, snapshot_data)
                VALUES (?,?,?,?,?,?)""",
                (
                    datetime.utcnow().isoformat(),
                    total_docs, stale_docs, avg_freshness,
                    int(critical_stale), json.dumps(data),
                ),
            )

    def resolve_alert(self, alert_id: str, resolved_by: str):
        with self._conn() as conn:
            conn.execute(
                """UPDATE clinical_alerts
                SET resolved=1, resolved_by=?, resolved_at=?
                WHERE alert_id=?""",
                (resolved_by, datetime.utcnow().isoformat(), alert_id),
            )

    def get_recent_results(self, limit: int = 50) -> List[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM monitoring_results ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_active_alerts(self) -> List[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM clinical_alerts WHERE resolved=0 ORDER BY timestamp DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_safety_summary(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM monitoring_results").fetchone()[0]
            passed = conn.execute(
                "SELECT COUNT(*) FROM monitoring_results WHERE overall_passed=1"
            ).fetchone()[0]
            critical_alerts = conn.execute(
                "SELECT COUNT(*) FROM clinical_alerts WHERE severity='critical' AND resolved=0"
            ).fetchone()[0]
            escalations = conn.execute(
                "SELECT COUNT(*) FROM clinical_alerts WHERE escalate_to_clinician=1 AND resolved=0"
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT alert_type, COUNT(*) FROM clinical_alerts GROUP BY alert_type"
            ).fetchall()
            avg_scores = conn.execute(
                """SELECT
                    AVG(retrieval_quality_score),
                    AVG(faithfulness_score),
                    AVG(guideline_freshness_score),
                    AVG(overall_safety_score)
                FROM monitoring_results"""
            ).fetchone()
            return {
                "total_queries_monitored": total,
                "queries_passed": passed,
                "pass_rate": round(passed / total, 3) if total > 0 else 0,
                "active_critical_alerts": critical_alerts,
                "pending_clinician_escalations": escalations,
                "alerts_by_type": {row[0]: row[1] for row in by_type},
                "avg_scores": {
                    "retrieval": round(avg_scores[0] or 0, 3),
                    "faithfulness": round(avg_scores[1] or 0, 3),
                    "freshness": round(avg_scores[2] or 0, 3),
                    "overall": round(avg_scores[3] or 0, 3),
                },
            }
