import re
from typing import List
from monitor.models import ClinicalDocument


MEDICAL_CLAIM_PATTERNS = [
    r'\b\d+\s*(?:mg|mcg|ml|units?|tablets?|capsules?)\b',
    r'\b(?:first[- ]line|second[- ]line|preferred|recommended|contraindicated)\b',
    r'\b(?:increases?|decreases?|reduces?|improves?|worsens?)\s+\w+\s+(?:risk|outcome|mortality|survival)\b',
    r'\b(?:FDA[- ]approved|guideline[- ]recommended|evidence[- ]based)\b',
    r'\b(?:do not|avoid|never|always|must)\s+(?:use|give|administer|prescribe)\b',
    r'\b\d+\s*(?:percent|%)\s+(?:of patients|efficacy|reduction|increase)\b',
]

HALLUCINATION_SIGNALS = [
    "studies show",
    "research indicates",
    "evidence suggests",
    "it is well known",
    "clinical trials have shown",
    "according to guidelines",
    "the standard of care",
    "it has been proven",
]

UNCERTAINTY_PHRASES = [
    "i'm not sure",
    "i cannot confirm",
    "i do not have",
    "this may vary",
    "consult a specialist",
    "i cannot provide",
]


def compute_faithfulness_score(answer: str, docs: List[ClinicalDocument]) -> dict:
    if not docs:
        return {
            "score": 0.0,
            "reason": "No source documents to verify against",
            "grounded_claims": 0,
            "ungrounded_claims": 0,
            "hallucination_risk": "high",
        }

    combined_source = " ".join(doc.content.lower() for doc in docs)
    answer_lower = answer.lower()

    medical_claims = []
    for pattern in MEDICAL_CLAIM_PATTERNS:
        matches = re.findall(pattern, answer_lower, re.IGNORECASE)
        medical_claims.extend(matches)

    grounded = 0
    ungrounded = 0
    for claim in medical_claims:
        claim_clean = claim.lower().strip()
        if claim_clean in combined_source:
            grounded += 1
        else:
            ungrounded += 1

    hallucination_signal_count = sum(
        1 for phrase in HALLUCINATION_SIGNALS
        if phrase in answer_lower
    )

    has_uncertainty = any(phrase in answer_lower for phrase in UNCERTAINTY_PHRASES)

    answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]
    grounded_sentences = 0
    for sentence in answer_sentences:
        sentence_words = set(sentence.lower().split())
        for doc in docs:
            doc_words = set(doc.content.lower().split())
            overlap = len(sentence_words & doc_words) / max(len(sentence_words), 1)
            if overlap > 0.3:
                grounded_sentences += 1
                break

    sentence_grounding = (
        grounded_sentences / len(answer_sentences)
        if answer_sentences else 0.5
    )

    total_claims = grounded + ungrounded
    claim_grounding = grounded / total_claims if total_claims > 0 else 0.7

    score = (sentence_grounding * 0.5) + (claim_grounding * 0.3)

    if hallucination_signal_count > 2:
        score -= 0.15
    if has_uncertainty:
        score -= 0.1
    if ungrounded > 2:
        score -= 0.1

    score = max(0.0, min(1.0, score))

    if score >= 0.7:
        hallucination_risk = "low"
        reason = "Answer appears well grounded in source documents"
    elif score >= 0.5:
        hallucination_risk = "medium"
        reason = f"Some claims may not be fully supported by retrieved documents ({ungrounded} ungrounded claims detected)"
    else:
        hallucination_risk = "high"
        reason = f"Answer contains claims not found in source documents. High hallucination risk. ({ungrounded} ungrounded claims)"

    return {
        "score": round(score, 3),
        "reason": reason,
        "grounded_claims": grounded,
        "ungrounded_claims": ungrounded,
        "hallucination_risk": hallucination_risk,
        "sentence_grounding": round(sentence_grounding, 3),
        "hallucination_signals_detected": hallucination_signal_count,
    }
