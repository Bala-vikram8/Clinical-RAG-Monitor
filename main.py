import argparse
import json
import uvicorn
from monitor.engine import RAGMonitor
from monitor.models import RAGQuery, RAGResponse, ClinicalDocument
from monitor.freshness_checker import get_kb_freshness_report
from alerts.audit_logger import AuditLogger
from rag.pipeline import CLINICAL_KNOWLEDGE_BASE, retrieve_documents


DEMO_QUERIES = [
    {
        "query": "What is the recommended first-line treatment for Type 2 diabetes in a patient with chronic kidney disease and eGFR of 35?",
        "specialty": "endocrinology",
        "patient_context": "65 year old male, T2DM x 10 years, CKD stage 3b, eGFR 35, no cardiovascular disease",
        "doctor_id": "DR-001",
    },
    {
        "query": "What blood pressure target should I aim for in a diabetic patient and which antihypertensive class is preferred?",
        "specialty": "cardiology",
        "patient_context": "58 year old female, T2DM, hypertension, no CKD, no cardiovascular events",
        "doctor_id": "DR-002",
    },
    {
        "query": "Patient presenting with fever, hypotension and elevated lactate. What is the management protocol?",
        "specialty": "infectious_disease",
        "patient_context": "72 year old male, nursing home resident, fever 39.2C, BP 85/50, lactate 4.2 mmol/L",
        "doctor_id": "DR-003",
    },
]


def print_monitoring_result(result, query_text: str):
    print(f"\n{'='*70}")
    print(f"MONITORING RESULT")
    print(f"{'='*70}")
    print(f"Query: {query_text[:65]}...")
    print(f"\nSCORES:")
    print(f"  Retrieval Quality:    {result.retrieval_quality_score:.3f} {'PASS' if result.retrieval_passed else 'FAIL'}")
    print(f"  Faithfulness:         {result.faithfulness_score:.3f} {'PASS' if result.faithfulness_passed else 'FAIL'}")
    print(f"  Guideline Freshness:  {result.guideline_freshness_score:.3f} {'PASS' if result.freshness_passed else 'FAIL'}")
    print(f"  Overall Safety:       {result.overall_safety_score:.3f} {'SAFE' if result.overall_passed else 'UNSAFE'}")

    if result.alerts:
        print(f"\nALERTS ({len(result.alerts)}):")
        for alert in result.alerts:
            print(f"  [{alert.severity.value.upper()}] {alert.alert_type.value}")
            print(f"  {alert.message}")
            if alert.escalate_to_clinician:
                print(f"  *** ESCALATE TO SENIOR CLINICIAN ***")

    print(f"\nRECOMMENDATIONS:")
    for rec in result.recommendations:
        print(f"  - {rec}")


def run_demo():
    print("\n" + "="*70)
    print("HEALTHCARE RAG MONITOR: Demo Mode (No API Calls)")
    print("="*70)

    monitor = RAGMonitor()
    audit = AuditLogger()

    for demo in DEMO_QUERIES:
        query = RAGQuery(
            doctor_id=demo["doctor_id"],
            patient_context=demo["patient_context"],
            query=demo["query"],
            specialty=demo["specialty"],
        )

        docs, scores = retrieve_documents(query.query, query.specialty)

        simulated_answer = f"""Based on the retrieved clinical guidelines, here is the evidence-based recommendation.

According to the guidelines, the recommended approach for this patient involves careful consideration
of the patient's renal function and comorbidities. The first-line treatment should be selected
based on the current guidelines which recommend specific medication classes for patients
with these characteristics. Regular monitoring of HbA1c and renal function is essential.
The target values should be individualized based on patient factors."""

        response = RAGResponse(
            query_id=query.query_id,
            retrieved_docs=docs,
            generated_answer=simulated_answer,
            retrieval_scores=scores,
            model_used="simulated",
        )

        result = monitor.monitor(query, response)
        audit.log_result(result)
        print_monitoring_result(result, query.query)

    print(f"\n{'='*70}")
    print("KNOWLEDGE BASE HEALTH CHECK")
    print(f"{'='*70}")
    kb_report = get_kb_freshness_report(CLINICAL_KNOWLEDGE_BASE)
    print(f"Total documents:    {len(CLINICAL_KNOWLEDGE_BASE)}")
    print(f"Stale documents:    {len(kb_report.get('stale_docs', []))}")
    print(f"Freshness score:    {kb_report['score']:.3f}")
    print(f"Oldest doc (days):  {kb_report.get('oldest_doc_days', 'N/A')}")
    if kb_report.get("stale_docs"):
        print("\nSTALE DOCUMENTS:")
        for doc in kb_report["stale_docs"]:
            print(f"  - {doc['title']} ({doc['days_old']} days old)")


def run_pipeline(query_text: str, specialty: str = "general"):
    from rag.pipeline import ClinicalRAGPipeline
    print(f"\n[PIPELINE] Running clinical query through RAG + Monitor...")
    pipeline = ClinicalRAGPipeline()
    monitor = RAGMonitor()
    audit = AuditLogger()

    query = RAGQuery(
        doctor_id="DR-CLI",
        patient_context="Provided via CLI",
        query=query_text,
        specialty=specialty,
    )

    response = pipeline.run(query)
    print(f"\n[RESPONSE]\n{response.generated_answer}")

    result = monitor.monitor(query, response)
    audit.log_result(result)
    print_monitoring_result(result, query_text)


def show_safety_report():
    audit = AuditLogger()
    report = audit.get_safety_summary()
    print("\n" + "="*70)
    print("SAFETY REPORT")
    print("="*70)
    print(json.dumps(report, indent=2))


def show_active_alerts():
    audit = AuditLogger()
    alerts = audit.get_active_alerts()
    print(f"\nActive alerts: {len(alerts)}")
    for a in alerts:
        print(f"  [{a['severity'].upper()}] {a['alert_type']} - {a['message'][:60]}")
        if a['escalate_to_clinician']:
            print(f"  *** REQUIRES CLINICIAN REVIEW ***")


def start_dashboard():
    print("\nStarting dashboard at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    uvicorn.run("dashboard.api:app", host="0.0.0.0", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(description="Healthcare RAG Monitor CLI")
    parser.add_argument(
        "mode",
        choices=["demo", "run", "report", "alerts", "kb-health", "dashboard"],
        help=(
            "demo: run monitor on sample clinical queries without API calls | "
            "run: run full pipeline with real API call | "
            "report: show safety report | "
            "alerts: show active alerts | "
            "kb-health: check knowledge base freshness | "
            "dashboard: start monitoring API"
        ),
    )
    parser.add_argument("--query", type=str, help="Clinical query for run mode")
    parser.add_argument("--specialty", type=str, default="general", help="Medical specialty")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "run":
        query = args.query or "What is the recommended treatment for hypertension in a diabetic patient?"
        run_pipeline(query, args.specialty)
    elif args.mode == "report":
        show_safety_report()
    elif args.mode == "alerts":
        show_active_alerts()
    elif args.mode == "kb-health":
        report = get_kb_freshness_report(CLINICAL_KNOWLEDGE_BASE)
        print(json.dumps(report, indent=2))
    elif args.mode == "dashboard":
        start_dashboard()


if __name__ == "__main__":
    main()
