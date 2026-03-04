from datetime import datetime, timedelta
from typing import List
from monitor.models import ClinicalDocument


SPECIALTY_UPDATE_CYCLES = {
    "cardiology": 365,
    "oncology": 180,
    "infectious_disease": 90,
    "endocrinology": 365,
    "nephrology": 365,
    "neurology": 365,
    "general": 730,
    "default": 365,
}

CRITICAL_SPECIALTIES = {
    "oncology",
    "infectious_disease",
    "cardiology",
}


def _parse_date(date_str: str) -> datetime | None:
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _days_since(date_str: str) -> int | None:
    dt = _parse_date(date_str)
    if not dt:
        return None
    return (datetime.utcnow() - dt).days


def compute_freshness_score(docs: List[ClinicalDocument]) -> dict:
    if not docs:
        return {
            "score": 0.0,
            "reason": "No documents to evaluate",
            "stale_docs": [],
            "newest_doc_days": None,
            "oldest_doc_days": None,
        }

    stale_docs = []
    doc_ages = []

    for doc in docs:
        days = _days_since(doc.last_updated)
        if days is None:
            days = _days_since(doc.published_date)

        if days is not None:
            doc_ages.append(days)
            specialty = doc.specialty.lower()
            max_age = SPECIALTY_UPDATE_CYCLES.get(
                specialty,
                SPECIALTY_UPDATE_CYCLES["default"]
            )

            if days > max_age:
                stale_docs.append({
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "days_old": days,
                    "max_acceptable_days": max_age,
                    "specialty": doc.specialty,
                    "guideline_version": doc.guideline_version,
                    "is_critical_specialty": specialty in CRITICAL_SPECIALTIES,
                })

    if not doc_ages:
        return {
            "score": 0.5,
            "reason": "Unable to determine document ages from metadata",
            "stale_docs": [],
            "newest_doc_days": None,
            "oldest_doc_days": None,
        }

    newest = min(doc_ages)
    oldest = max(doc_ages)
    stale_ratio = len(stale_docs) / len(docs)
    has_critical_stale = any(s["is_critical_specialty"] for s in stale_docs)

    score = 1.0 - (stale_ratio * 0.7)

    if oldest > 1825:
        score -= 0.2
    elif oldest > 730:
        score -= 0.1

    if has_critical_stale:
        score -= 0.2

    score = max(0.0, min(1.0, score))

    if stale_docs:
        reason = (
            f"{len(stale_docs)} of {len(docs)} documents are outdated. "
            f"Oldest document is {oldest} days old."
        )
        if has_critical_stale:
            reason += " CRITICAL: Outdated guidelines in high-risk specialty detected."
    else:
        reason = f"All documents within acceptable freshness window. Newest: {newest} days old."

    return {
        "score": round(score, 3),
        "reason": reason,
        "stale_docs": stale_docs,
        "newest_doc_days": newest,
        "oldest_doc_days": oldest,
        "stale_ratio": round(stale_ratio, 3),
        "has_critical_stale": has_critical_stale,
    }


def get_kb_freshness_report(docs: List[ClinicalDocument]) -> dict:
    result = compute_freshness_score(docs)
    specialty_breakdown = {}
    for doc in docs:
        sp = doc.specialty
        if sp not in specialty_breakdown:
            specialty_breakdown[sp] = {"total": 0, "stale": 0}
        specialty_breakdown[sp]["total"] += 1

    stale_ids = {s["doc_id"] for s in result["stale_docs"]}
    for doc in docs:
        if doc.doc_id in stale_ids:
            specialty_breakdown[doc.specialty]["stale"] += 1

    return {**result, "specialty_breakdown": specialty_breakdown}
