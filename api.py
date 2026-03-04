from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from monitor.engine import RAGMonitor
from monitor.models import RAGQuery, RAGResponse, ClinicalDocument
from monitor.freshness_checker import get_kb_freshness_report
from alerts.audit_logger import AuditLogger
from rag.pipeline import CLINICAL_KNOWLEDGE_BASE

app = FastAPI(
    title="Healthcare RAG Monitor",
    description="Clinical RAG pipeline drift detection and compliance audit system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

monitor = RAGMonitor()
audit = AuditLogger()


class MonitorRequest(BaseModel):
    query: dict
    response: dict


@app.get("/")
def root():
    return {"service": "Healthcare RAG Monitor", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/safety/summary")
def safety_summary():
    return audit.get_safety_summary()


@app.get("/alerts/active")
def active_alerts():
    return {"alerts": audit.get_active_alerts()}


@app.post("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str, resolved_by: str = "admin"):
    audit.resolve_alert(alert_id, resolved_by)
    return {"status": "resolved", "alert_id": alert_id}


@app.get("/results/recent")
def recent_results(limit: int = 20):
    return {"results": audit.get_recent_results(limit=limit)}


@app.get("/kb/health")
def kb_health():
    report = get_kb_freshness_report(CLINICAL_KNOWLEDGE_BASE)
    audit.log_kb_snapshot(
        total_docs=len(CLINICAL_KNOWLEDGE_BASE),
        stale_docs=len(report.get("stale_docs", [])),
        avg_freshness=report.get("score", 0),
        critical_stale=report.get("has_critical_stale", False),
        data=report,
    )
    return report


@app.get("/kb/documents")
def list_documents():
    return {
        "total": len(CLINICAL_KNOWLEDGE_BASE),
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "specialty": d.specialty,
                "guideline_version": d.guideline_version,
                "last_updated": d.last_updated,
                "source": d.source,
            }
            for d in CLINICAL_KNOWLEDGE_BASE
        ],
    }
