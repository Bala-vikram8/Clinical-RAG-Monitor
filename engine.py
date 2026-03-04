from monitor.models import (
    MonitoringResult, ClinicalAlert, AlertType, AlertSeverity,
    RAGQuery, RAGResponse,
)
from monitor.retrieval_checker import compute_retrieval_score, check_specialty_alignment
from monitor.faithfulness_checker import compute_faithfulness_score
from monitor.freshness_checker import compute_freshness_score


THRESHOLDS = {
    "retrieval_quality": 0.55,
    "faithfulness": 0.60,
    "guideline_freshness": 0.65,
    "overall_safety": 0.60,
}


def _severity_from_score(score: float, threshold: float) -> AlertSeverity:
    gap = threshold - score
    if gap > 0.3:
        return AlertSeverity.CRITICAL
    elif gap > 0.2:
        return AlertSeverity.HIGH
    elif gap > 0.1:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


def _build_retrieval_alert(query_id: str, result: dict, score: float) -> ClinicalAlert:
    severity = _severity_from_score(score, THRESHOLDS["retrieval_quality"])
    return ClinicalAlert(
        query_id=query_id,
        alert_type=AlertType.LOW_RETRIEVAL_QUALITY,
        severity=severity,
        message=f"Retrieved documents may not be relevant to this clinical query. Score: {score:.2f}",
        details=result,
        action_required="Review retrieved documents manually before acting on this response",
        escalate_to_clinician=severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL),
    )


def _build_faithfulness_alert(query_id: str, result: dict, score: float) -> ClinicalAlert:
    severity = _severity_from_score(score, THRESHOLDS["faithfulness"])
    return ClinicalAlert(
        query_id=query_id,
        alert_type=AlertType.HALLUCINATION,
        severity=severity,
        message=(
            f"Generated answer contains claims not fully supported by source documents. "
            f"Hallucination risk: {result.get('hallucination_risk', 'unknown')}. Score: {score:.2f}"
        ),
        details=result,
        action_required="Do not act on this response without manual verification by a clinician",
        escalate_to_clinician=True,
    )


def _build_freshness_alert(query_id: str, result: dict, score: float) -> ClinicalAlert:
    severity = _severity_from_score(score, THRESHOLDS["guideline_freshness"])
    stale_count = len(result.get("stale_docs", []))
    return ClinicalAlert(
        query_id=query_id,
        alert_type=AlertType.OUTDATED_GUIDELINE,
        severity=severity,
        message=(
            f"{stale_count} outdated clinical guidelines detected in knowledge base. "
            f"Freshness score: {score:.2f}"
        ),
        details={
            "stale_docs": result.get("stale_docs", []),
            "oldest_doc_days": result.get("oldest_doc_days"),
            "has_critical_stale": result.get("has_critical_stale", False),
        },
        action_required=(
            "Update knowledge base with current clinical guidelines before relying on this system. "
            "Flag affected specialties for immediate review."
        ),
        escalate_to_clinician=result.get("has_critical_stale", False),
    )


def _build_recommendations(
    retrieval_ok: bool,
    faithfulness_ok: bool,
    freshness_ok: bool,
    retrieval_result: dict,
    faithfulness_result: dict,
    freshness_result: dict,
) -> list[str]:
    recs = []
    if not retrieval_ok:
        recs.append(
            f"Improve retrieval: Only {retrieval_result['doc_count']} documents retrieved "
            f"with {retrieval_result['keyword_overlap']:.0%} keyword overlap. "
            "Consider expanding the vector search radius or improving query preprocessing."
        )
    if not faithfulness_ok:
        ungrounded = faithfulness_result.get("ungrounded_claims", 0)
        recs.append(
            f"Reduce hallucination: {ungrounded} medical claims in the answer are not "
            "traceable to retrieved documents. Add a citation verification step before delivery."
        )
    if not freshness_ok:
        stale = freshness_result.get("stale_docs", [])
        specialties = list({s["specialty"] for s in stale})
        recs.append(
            f"Update knowledge base: {len(stale)} documents are outdated across "
            f"specialties: {', '.join(specialties)}. Schedule an immediate KB refresh."
        )
    if not recs:
        recs.append("All monitoring checks passed. Response is safe to deliver to clinician.")
    return recs


class RAGMonitor:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or THRESHOLDS

    def monitor(self, query: RAGQuery, response: RAGResponse) -> MonitoringResult:
        retrieval_result = compute_retrieval_score(
            query.query,
            response.retrieved_docs,
            response.retrieval_scores,
        )
        faithfulness_result = compute_faithfulness_score(
            response.generated_answer,
            response.retrieved_docs,
        )
        freshness_result = compute_freshness_score(response.retrieved_docs)

        specialty_alignment = check_specialty_alignment(
            query.specialty,
            response.retrieved_docs,
        )

        r_score = retrieval_result["score"]
        f_score = faithfulness_result["score"]
        fr_score = freshness_result["score"]

        overall = (r_score * 0.35) + (f_score * 0.40) + (fr_score * 0.25)

        r_pass = r_score >= self.thresholds["retrieval_quality"]
        f_pass = f_score >= self.thresholds["faithfulness"]
        fr_pass = fr_score >= self.thresholds["guideline_freshness"]
        overall_pass = overall >= self.thresholds["overall_safety"]

        alerts = []
        if not r_pass:
            alerts.append(_build_retrieval_alert(query.query_id, retrieval_result, r_score))
        if not f_pass:
            alerts.append(_build_faithfulness_alert(query.query_id, faithfulness_result, f_score))
        if not fr_pass:
            alerts.append(_build_freshness_alert(query.query_id, freshness_result, fr_score))

        recommendations = _build_recommendations(
            r_pass, f_pass, fr_pass,
            retrieval_result, faithfulness_result, freshness_result,
        )

        audit_trail = {
            "query": query.query,
            "doctor_id": query.doctor_id,
            "specialty": query.specialty,
            "model_used": response.model_used,
            "documents_retrieved": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "guideline_version": d.guideline_version,
                    "last_updated": d.last_updated,
                    "source": d.source,
                }
                for d in response.retrieved_docs
            ],
            "retrieval_detail": retrieval_result,
            "faithfulness_detail": faithfulness_result,
            "freshness_detail": freshness_result,
            "specialty_alignment": specialty_alignment,
            "thresholds_used": self.thresholds,
        }

        return MonitoringResult(
            query_id=query.query_id,
            response_id=response.response_id,
            retrieval_quality_score=round(r_score, 3),
            faithfulness_score=round(f_score, 3),
            guideline_freshness_score=round(fr_score, 3),
            overall_safety_score=round(overall, 3),
            retrieval_passed=r_pass,
            faithfulness_passed=f_pass,
            freshness_passed=fr_pass,
            overall_passed=overall_pass,
            alerts=alerts,
            recommendations=recommendations,
            audit_trail=audit_trail,
        )
