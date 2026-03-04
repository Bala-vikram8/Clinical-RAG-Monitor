# Healthcare RAG Monitor

A clinical RAG pipeline monitoring system for hospital clinical decision support.
Detects hallucinations, outdated clinical guidelines, and retrieval quality drift
before AI-generated answers reach doctors. Produces a full compliance audit trail
for every query.

---

## The Problem

Small hospitals and clinics are deploying RAG systems to help doctors make clinical
decisions faster. A doctor asks the system a question, it retrieves relevant clinical
guidelines, and generates an answer.

The problem is these systems degrade silently in three ways.

First, medical guidelines change. The ADA, CDC, and professional bodies update
treatment recommendations regularly. If your knowledge base is not current, doctors
get answers based on outdated guidelines. In medicine that can directly harm patients.

Second, the LLM starts hallucinating medical facts that were never in the retrieved
documents. The answer sounds confident and plausible but it is not grounded in any
source. A doctor acts on it. That is a patient safety incident.

Third, the retrieval drifts. The system starts pulling the wrong documents for a
given query and nobody notices because the answers still look reasonable.

None of these failures throw an error. The system keeps running. Doctors keep trusting
it. That is the danger.

---

## What This Monitors

Every RAG response goes through three checks before it is logged as safe or flagged.

**Retrieval Quality Check**
Measures whether the documents retrieved actually match the clinical query.
Uses keyword overlap and vector similarity scores. Flags responses where
the retrieved context is likely irrelevant to the question asked.

**Faithfulness Check**
Detects whether the generated answer is grounded in the retrieved documents
or whether the model is introducing facts not found in the sources.
Scans for medical claims like dosages, drug names, and treatment recommendations
and checks whether each claim can be traced back to the retrieved context.

**Guideline Freshness Check**
Checks the publication and update dates of every retrieved document against
specialty-specific freshness thresholds. Oncology guidelines have a 180 day
threshold. Infectious disease has 90 days. General medicine has 730 days.
Flags stale documents and marks critical specialties for immediate KB refresh.

---

## Architecture

```
healthcare-rag-monitor/
├── monitor/
│   ├── models.py               Data models for queries, responses, alerts
│   ├── retrieval_checker.py    Measures retrieval relevance and quality
│   ├── faithfulness_checker.py Detects hallucinations and ungrounded claims
│   ├── freshness_checker.py    Checks clinical guideline publication dates
│   └── engine.py               Core monitor that runs all three checks
├── alerts/
│   └── audit_logger.py         SQLite audit trail for compliance logging
├── rag/
│   └── pipeline.py             Simulated clinical RAG pipeline with knowledge base
├── dashboard/
│   └── api.py                  FastAPI monitoring and audit API
├── main.py                     CLI entry point
└── config.py                   Environment configuration
```

---

## Setup

**Prerequisites**
- Python 3.11+
- Anthropic API key

**Installation**

```bash
git clone https://github.com/yourusername/healthcare-rag-monitor
cd healthcare-rag-monitor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Anthropic API key to `.env`.

---

## Running It

**Demo mode, no API calls**
```bash
python main.py demo
```
Runs three clinical queries through the monitor using simulated responses.
Shows monitoring scores, alerts, and the KB health report. No API key needed.

**Run a real clinical query through the full pipeline**
```bash
python main.py run --query "What is the first-line treatment for T2 diabetes with CKD?" --specialty endocrinology
```

**View the safety report**
```bash
python main.py report
```

**View active clinical alerts**
```bash
python main.py alerts
```

**Check knowledge base freshness**
```bash
python main.py kb-health
```

**Start the monitoring dashboard**
```bash
python main.py dashboard
```
Opens at http://localhost:8000. Full API docs at http://localhost:8000/docs.

---

## Dashboard API

| Method | Endpoint | What it does |
|---|---|---|
| GET | /safety/summary | Overall safety scores and pass rates |
| GET | /alerts/active | All unresolved clinical alerts |
| POST | /alerts/{id}/resolve | Mark an alert as resolved |
| GET | /results/recent | Recent monitoring results |
| GET | /kb/health | Knowledge base freshness report |
| GET | /kb/documents | List all documents with metadata |

---

## Clinical Alert Types

| Alert | Severity | Action |
|---|---|---|
| hallucination_detected | High to Critical | Escalate to senior clinician. Do not act on response. |
| outdated_guideline | Medium to Critical | Refresh knowledge base. Flag specialty for review. |
| low_retrieval_quality | Low to High | Review retrieved documents manually. |

---

## Why This Matters for Hospitals

Joint Commission and CMS standards require hospitals to document the basis for
clinical decision support recommendations. If a doctor acts on a RAG answer
based on a 2014 guideline superseded in 2022, the hospital has a liability problem.

This system produces a structured audit trail for every single query showing
which guideline versions were used, what the retrieval quality was, whether
the answer was flagged before reaching the doctor, and who resolved the alert.
That is not just an engineering feature. It is a compliance feature that
hospital administrators and risk management teams need.

---

## Tech Stack

Python 3.11, Anthropic Claude API, FastAPI, Pydantic v2, SQLite, Uvicorn

---

## Contributing

Priority areas: integration with real vector databases like Pinecone or Weaviate,
LLM-based faithfulness judge to replace the heuristic checker, automated KB refresh
pipeline when stale documents are detected, and HL7 FHIR compatibility for EHR integration.
