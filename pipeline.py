import anthropic
import random
from typing import List
from monitor.models import ClinicalDocument, RAGQuery, RAGResponse
from config import ANTHROPIC_API_KEY, MODEL


CLINICAL_KNOWLEDGE_BASE = [
    ClinicalDocument(
        doc_id="guideline-dm2-001",
        title="ADA Standards of Medical Care in Diabetes 2024",
        content="""Type 2 Diabetes Management Guidelines 2024.
        First-line therapy: Metformin remains the preferred initial pharmacologic agent for type 2 diabetes
        unless contraindicated. For patients with established cardiovascular disease, heart failure, or
        chronic kidney disease (CKD), GLP-1 receptor agonists or SGLT2 inhibitors with proven cardiovascular
        benefit are recommended as first-line or add-on therapy regardless of HbA1c.
        For CKD patients: SGLT2 inhibitors (empagliflozin, dapagliflozin, canagliflozin) are recommended
        to reduce CKD progression and cardiovascular risk. Avoid metformin if eGFR < 30.
        HbA1c targets: Generally < 7% for most adults. Less stringent target (< 8%) acceptable for
        patients with limited life expectancy or high hypoglycemia risk.
        Blood pressure target: < 130/80 mmHg for most patients with diabetes and hypertension.
        Monitoring: HbA1c every 3 months until stable, then every 6 months.""",
        source="American Diabetes Association",
        guideline_version="2024.1",
        published_date="2024-01-01",
        last_updated="2024-01-15",
        specialty="endocrinology",
        icd_codes=["E11", "E11.9", "E11.65"],
    ),
    ClinicalDocument(
        doc_id="guideline-htn-001",
        title="JNC 8 Hypertension Guidelines",
        content="""Hypertension Management - JNC 8 Guidelines.
        Blood pressure thresholds for initiating treatment:
        General population age >= 60: BP >= 150/90 mmHg. Treat to < 150/90.
        General population age < 60: DBP >= 90 mmHg. Treat to < 90.
        Diabetes or CKD patients: BP >= 140/90. Treat to < 140/90.
        First-line medications: Thiazide diuretics, calcium channel blockers (CCBs),
        ACE inhibitors, ARBs. Non-black patients including diabetics: ACE inhibitor or ARB preferred.
        Black patients: Thiazide or CCB preferred.
        CKD patients: ACE inhibitor or ARB recommended regardless of race.""",
        source="Joint National Committee",
        guideline_version="JNC-8",
        published_date="2014-02-05",
        last_updated="2014-02-05",
        specialty="cardiology",
        icd_codes=["I10", "I11", "I12"],
    ),
    ClinicalDocument(
        doc_id="guideline-ckd-001",
        title="KDIGO CKD Management Guidelines 2022",
        content="""Chronic Kidney Disease Management - KDIGO 2022.
        CKD staging based on GFR and albuminuria categories.
        G1: GFR >= 90. G2: GFR 60-89. G3a: GFR 45-59. G3b: GFR 30-44. G4: GFR 15-29. G5: GFR < 15.
        Blood pressure target in CKD: < 120 mmHg systolic using standardized measurement.
        ACE inhibitors or ARBs: Recommended in CKD with albuminuria > 300 mg/g to slow progression.
        SGLT2 inhibitors: Recommended in CKD with eGFR >= 20 and albuminuria > 200 mg/g.
        Avoid: NSAIDs, nephrotoxic medications. Adjust drug doses for reduced GFR.
        Referral to nephrology: eGFR < 30, rapidly declining GFR, or complex management needs.""",
        source="Kidney Disease Improving Global Outcomes",
        guideline_version="KDIGO-2022",
        published_date="2022-06-01",
        last_updated="2022-08-01",
        specialty="nephrology",
        icd_codes=["N18", "N18.3", "N18.4", "N18.5"],
    ),
    ClinicalDocument(
        doc_id="guideline-sepsis-001",
        title="Surviving Sepsis Campaign Guidelines 2021",
        content="""Sepsis and Septic Shock Management - Surviving Sepsis 2021.
        Hour-1 Bundle: Measure lactate. Obtain blood cultures before antibiotics.
        Administer broad-spectrum antibiotics within 1 hour of recognition.
        Administer 30 ml/kg crystalloid for hypotension or lactate >= 4 mmol/L.
        Apply vasopressors if hypotensive during or after fluid resuscitation.
        Norepinephrine: First-line vasopressor for septic shock. Target MAP >= 65 mmHg.
        Vasopressin: Add-on if norepinephrine requirements are high.
        Hydrocortisone: Consider if hemodynamically unstable despite adequate fluids and vasopressors.
        Antibiotic de-escalation: Reassess daily based on cultures and clinical response.""",
        source="Society of Critical Care Medicine",
        guideline_version="SSC-2021",
        published_date="2021-10-01",
        last_updated="2021-10-15",
        specialty="infectious_disease",
        icd_codes=["A41", "A41.9", "R65.20", "R65.21"],
    ),
    ClinicalDocument(
        doc_id="guideline-htn-old-001",
        title="JNC 7 Hypertension Guidelines (SUPERSEDED)",
        content="""Hypertension Management - JNC 7 (2003). OUTDATED - Superseded by JNC 8.
        Prehypertension: SBP 120-139 or DBP 80-89. Lifestyle modification recommended.
        Stage 1: SBP 140-159 or DBP 90-99. Thiazide diuretics for most patients.
        Stage 2: SBP >= 160 or DBP >= 100. Two-drug combination usually required.
        Thiazide diuretics: Preferred for most patients unless contraindicated.
        Compelling indications may require specific drug classes.""",
        source="Joint National Committee",
        guideline_version="JNC-7",
        published_date="2003-05-21",
        last_updated="2003-05-21",
        specialty="cardiology",
        icd_codes=["I10"],
    ),
]


def retrieve_documents(query: str, specialty: str, top_k: int = 3) -> tuple[List[ClinicalDocument], List[float]]:
    query_lower = query.lower()
    scored = []

    for doc in CLINICAL_KNOWLEDGE_BASE:
        score = 0.0
        doc_text = (doc.content + " " + doc.title).lower()

        query_words = set(query_lower.split())
        doc_words = set(doc_text.split())
        overlap = len(query_words & doc_words) / max(len(query_words), 1)
        score += overlap * 0.6

        if doc.specialty.lower() == specialty.lower():
            score += 0.3
        elif doc.specialty == "general":
            score += 0.1

        score += random.uniform(0, 0.1)
        scored.append((doc, min(1.0, score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    return [d for d, _ in top], [round(s, 3) for _, s in top]


class ClinicalRAGPipeline:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def run(self, query: RAGQuery) -> RAGResponse:
        docs, scores = retrieve_documents(query.query, query.specialty)

        context = "\n\n".join([
            f"[SOURCE: {doc.title} | Version: {doc.guideline_version} | Updated: {doc.last_updated}]\n{doc.content}"
            for doc in docs
        ])

        system_prompt = """You are a clinical decision support assistant for a hospital system.
Your role is to assist doctors by providing evidence-based clinical information.

CRITICAL RULES:
1. Only use information from the provided clinical guidelines. Do not add information not in the sources.
2. Always cite the specific guideline you are referencing.
3. Flag any uncertainty explicitly.
4. Never recommend a specific treatment without noting that clinical judgment is required.
5. If guidelines are outdated or conflicting, state this explicitly."""

        user_message = f"""Clinical Query from Doctor:
{query.query}

Patient Context:
{query.patient_context}

Relevant Clinical Guidelines:
{context}

Please provide a concise, evidence-based response citing the specific guidelines used."""

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return RAGResponse(
            query_id=query.query_id,
            retrieved_docs=docs,
            generated_answer=response.content[0].text,
            retrieval_scores=scores,
            model_used=MODEL,
        )
