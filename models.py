from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    HALLUCINATION = "hallucination_detected"
    OUTDATED_GUIDELINE = "outdated_guideline"
    LOW_RETRIEVAL_QUALITY = "low_retrieval_quality"
    FAITHFULNESS_DRIFT = "faithfulness_drift"
    KNOWLEDGE_BASE_STALE = "knowledge_base_stale"


class ClinicalDocument(BaseModel):
    doc_id: str = ""
    title: str
    content: str
    source: str
    guideline_version: str
    published_date: str
    last_updated: str
    specialty: str
    icd_codes: List[str] = []

    def model_post_init(self, __context):
        if not self.doc_id:
            self.doc_id = str(uuid.uuid4())


class RAGQuery(BaseModel):
    query_id: str = ""
    timestamp: str = ""
    doctor_id: str
    patient_context: str
    query: str
    specialty: str

    def model_post_init(self, __context):
        if not self.query_id:
            self.query_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class RAGResponse(BaseModel):
    response_id: str = ""
    query_id: str
    timestamp: str = ""
    retrieved_docs: List[ClinicalDocument]
    generated_answer: str
    retrieval_scores: List[float]
    model_used: str

    def model_post_init(self, __context):
        if not self.response_id:
            self.response_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class MonitoringResult(BaseModel):
    result_id: str = ""
    timestamp: str = ""
    query_id: str
    response_id: str

    retrieval_quality_score: float
    faithfulness_score: float
    guideline_freshness_score: float
    overall_safety_score: float

    retrieval_passed: bool
    faithfulness_passed: bool
    freshness_passed: bool
    overall_passed: bool

    alerts: List["ClinicalAlert"] = []
    recommendations: List[str] = []
    audit_trail: Dict[str, Any] = {}

    def model_post_init(self, __context):
        if not self.result_id:
            self.result_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class ClinicalAlert(BaseModel):
    alert_id: str = ""
    timestamp: str = ""
    query_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = {}
    action_required: str
    escalate_to_clinician: bool = False

    def model_post_init(self, __context):
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


MonitoringResult.model_rebuild()
