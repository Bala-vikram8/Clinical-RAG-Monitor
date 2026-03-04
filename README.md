# Clinical RAG Monitor

Hospitals are using AI to help doctors make faster clinical decisions.
A doctor asks a question, the system pulls relevant medical guidelines,
and generates an answer. The problem is nobody is watching whether those
answers are actually safe to act on.

This project monitors every AI response in a clinical decision support
system and catches three specific problems before the answer reaches
a doctor.

One, the AI is making up medical facts that were never in the source
documents it retrieved. Two, the guidelines being used to generate the
answer are outdated and have been superseded by newer recommendations.
Three, the wrong documents are being retrieved for the question being
asked, so the answer is built on irrelevant context.

None of these failures throw an error. The system keeps running and
doctors keep trusting it. This monitor catches them.

---

## What Gets Checked on Every Query

**Retrieval Quality**
Checks whether the documents pulled from the knowledge base actually
match the clinical question being asked. Scores keyword overlap and
document relevance. Flags responses where the retrieved context is
likely off-topic.

**Hallucination Detection**
Scans the generated answer for medical claims like drug dosages,
treatment recommendations, and clinical targets. Checks whether each
claim can be traced back to the retrieved source documents. Flags
anything the AI added that was not in the sources.

**Guideline Freshness**
Checks the publication and update dates of every retrieved document.
Different medical specialties have different freshness thresholds.
Oncology guidelines expire in 180 days. Infectious disease in 90 days.
General medicine in 730 days. Stale documents trigger an immediate
alert and a knowledge base refresh recommendation.

---

## Setup

You need Python 3.11 or higher and an Anthropic API key.
```bash
git clone https://github.com/yourusername/clinical-rag-monitor
cd clinical-rag-monitor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open the `.env` file and add your Anthropic API key.
```
ANTHROPIC_API_KEY=your_key_here
```

---

## Running It

**See the monitor in action without any API calls**
```bash
python main.py demo
```
Runs three real clinical queries through all three checks. Shows
monitoring scores, clinical alerts, and a knowledge base health
report. No API key needed to run this.

**Run a real clinical query through the full pipeline**
```bash
python main.py run --query "First-line treatment for sepsis?" --specialty infectious_disease
```

**Check which guidelines in the knowledge base are outdated**
```bash
python main.py kb-health
```

**See all active clinical alerts that need attention**
```bash
python main.py alerts
```

**View the overall safety report**
```bash
python main.py report
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
| GET | /safety/summary | Overall pass rates and safety scores |
| GET | /alerts/active | All unresolved clinical alerts |
| POST | /alerts/{id}/resolve | Mark an alert as resolved |
| GET | /results/recent | Recent monitoring results |
| GET | /kb/health | Knowledge base freshness report |
| GET | /kb/documents | All documents with version and date info |

---

## Alert Types

| Alert | When it fires | What to do |
|---|---|---|
| hallucination_detected | AI answer contains ungrounded medical claims | Do not act on response. Escalate to senior clinician. |
| outdated_guideline | Retrieved document is past its freshness threshold | Refresh knowledge base. Flag the specialty. |
| low_retrieval_quality | Retrieved documents do not match the query | Review documents manually before acting. |

---

## Why This Matters for Hospitals

Joint Commission and CMS standards require hospitals to document the
basis for every clinical decision support recommendation. If a doctor
acts on an AI answer that was built on a 2014 guideline that was
superseded in 2022, the hospital has a liability problem.

This system writes a structured audit record for every single query.
It logs which guideline versions were used, what the retrieval quality
was, whether the answer was flagged before reaching the doctor, and
who reviewed and resolved each alert. That is not just an engineering
feature. It is a compliance feature that hospital risk management teams
actually need.

---

## Tech Stack

Python 3.11, Anthropic Claude API, FastAPI, Pydantic v2, SQLite, Uvicorn

---

## Contributing

Good areas to contribute: connecting to real vector databases like
Pinecone or Weaviate, building an LLM-based faithfulness judge to
replace the heuristic approach, automated knowledge base refresh when
stale documents are detected, and HL7 FHIR compatibility for EHR
integration.
