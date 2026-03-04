import re
from typing import List
from monitor.models import ClinicalDocument


MEDICAL_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "need", "dare", "ought", "used", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then",
    "once", "what", "which", "who", "when", "where", "why", "how",
    "this", "that", "these", "those", "i", "you", "he", "she",
    "it", "we", "they", "patient", "doctor", "treatment", "therapy",
}


def _extract_keywords(text: str) -> set:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in MEDICAL_STOP_WORDS}


def compute_retrieval_score(query: str, docs: List[ClinicalDocument], scores: List[float]) -> dict:
    if not docs:
        return {
            "score": 0.0,
            "reason": "No documents retrieved",
            "doc_count": 0,
            "keyword_overlap": 0.0,
            "avg_vector_score": 0.0,
        }

    avg_vector_score = sum(scores) / len(scores) if scores else 0.0

    query_keywords = _extract_keywords(query)
    overlaps = []
    for doc in docs:
        doc_keywords = _extract_keywords(doc.content + " " + doc.title)
        if query_keywords:
            overlap = len(query_keywords & doc_keywords) / len(query_keywords)
            overlaps.append(overlap)

    avg_keyword_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    score = (avg_vector_score * 0.6) + (avg_keyword_overlap * 0.4)

    reason = "Retrieval quality acceptable"
    if avg_vector_score < 0.5:
        reason = f"Low vector similarity scores (avg: {avg_vector_score:.2f})"
    elif avg_keyword_overlap < 0.3:
        reason = f"Low keyword overlap between query and retrieved docs ({avg_keyword_overlap:.2f})"
    elif len(docs) < 2:
        reason = "Only one document retrieved, limited context"

    return {
        "score": round(score, 3),
        "reason": reason,
        "doc_count": len(docs),
        "keyword_overlap": round(avg_keyword_overlap, 3),
        "avg_vector_score": round(avg_vector_score, 3),
    }


def check_specialty_alignment(query_specialty: str, docs: List[ClinicalDocument]) -> float:
    if not docs:
        return 0.0
    aligned = sum(
        1 for doc in docs
        if doc.specialty.lower() == query_specialty.lower()
        or doc.specialty.lower() == "general"
    )
    return round(aligned / len(docs), 3)
