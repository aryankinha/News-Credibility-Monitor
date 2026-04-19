# News Credibility Monitor

A hybrid ML + agentic reasoning system for news credibility analysis.

The project combines:
- A classical baseline: TF-IDF + Logistic Regression
- A multi-agent reasoning pipeline: LangGraph + Groq + NVIDIA judge + Chroma retrieval
- A FastAPI backend and a React/Vite frontend

## Current Status

- Milestone 1 (implemented): Classical ML credibility classification
- Milestone 2 (partially implemented): Multi-agent analysis with RAG and judge synthesis

## Highlights

- Dataset: ISOT Fake/True news CSV files
- Baseline model: Logistic Regression with TF-IDF bigrams
- Test performance (latest tracked):
  - Accuracy: 98.88%
  - Weighted Precision: 98.88%
  - Weighted Recall: 98.88%
  - Weighted F1: 98.88%
- Frontend UX includes animated pipeline stages, agent breakdown, evidence and risk factors

Metrics source in repo:
- `docs/metrics.json`

## Repository Layout

```text
News-Credibility-Monitor/
├── README.md
├── requirements_deploy.txt
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── data/raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   ├── models/
│   ├── scripts/build_embeddings.py
│   └── src/
│       ├── agent/
│       ├── config/
│       ├── data/
│       ├── features/
│       ├── llm/
│       ├── models/
│       ├── pipeline/
│       ├── rag/
│       └── utils/
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
├── docs/
├── notebooks/
└── Report/
```

## Backend Overview

Backend entrypoint:
- `backend/main.py`

API endpoints:
- `GET /` health/status
- `POST /analyze` full analysis pipeline

Input validation:
- Requires at least 50 words in `text`

Example request:

```json
{
  "text": "Your article text with at least 50 words..."
}
```

Example response shape:

```json
{
  "agent_a": { "verdict": "REAL", "confidence": "89", "reasoning": "..." },
  "agent_b": { "verdict": "FAKE", "confidence": "78", "reasoning": "..." },
  "agent_c": { "verdict": "REAL", "confidence": "84", "reasoning": "..." },
  "final": {
    "verdict": "REAL",
    "confidence": "86",
    "consensus": "...",
    "dominant_agent": "Agent A",
    "conflict": "..."
  },
  "agreement": {
    "level": "Medium",
    "distribution": { "REAL": 2, "FAKE": 1 }
  },
  "rag_summary": {
    "total_docs": 5,
    "real_docs": 3,
    "fake_docs": 2,
    "previews": ["...", "..."]
  },
  "risk_factors": ["..."],
  "ml_signal": "REAL (91.2%)",
  "rag_count": 5
}
```

Note:
- The backend currently returns error payloads as JSON objects, for example `{ "error": "..." }`.

## Agentic Pipeline

Core graph and nodes:
- `backend/src/agent/graph.py`
- `backend/src/agent/nodes.py`
- `backend/src/agent/state.py`

Execution flow:
1. `preprocess_node` cleans text
2. `ml_node` predicts class + confidence
3. Route by confidence threshold (85%):
   - High confidence: skip retrieval and continue to agents
   - Low confidence: run retrieval first
4. Agent A (conservative), Agent B (skeptical), Agent C (neutral)
5. Judge synthesizes consensus
6. Output node builds structured report

LLM providers in code:
- Primary agent generation: Groq API (`GROQ_API_KEY`)
- Judge attempt: NVIDIA endpoint (`NVIDIA_API_KEY`), with fallback

## ML Pipeline

Training orchestration:
- `backend/src/pipeline/training_pipeline.py`

Main stages:
1. Load and merge `Fake.csv` and `True.csv`
2. Clean text (dateline removal, regex cleanup, stopword removal)
3. Build TF-IDF features (`max_features=10000`, `ngram_range=(1,2)`)
4. Train Logistic Regression (`class_weight="balanced"`)
5. Evaluate and persist artifacts

Artifacts saved to:
- `backend/models/best_model.pkl`
- `backend/models/tfidf_vectorizer.pkl`
- `backend/models/metrics.json`
- `backend/models/confusion_matrix.png`

## Retrieval (RAG)

Implemented components:
- `backend/src/rag/build_db.py` (persistent Chroma DB build path)
- `backend/src/rag/load_embeddings.py` (load from precomputed embeddings)
- `backend/src/rag/retriever.py` (top-k retrieval)
- `backend/scripts/build_embeddings.py` (creates `models/embeddings.pkl`)

Embedding model used:
- `all-MiniLM-L6-v2`

## Frontend Overview

Frontend app entry:
- `frontend/src/App.jsx`

Stack:
- React + Vite + Tailwind + OGL background effect

UI behavior:
- Enforces minimum 50 words before submission
- Calls backend `POST /analyze`
- Shows staged pipeline animation
- Normalizes backend payload into:
  - final verdict
  - agreement block
  - per-agent cards
  - evidence summary
  - risk factors

Normalization logic:
- `frontend/src/lib/normalizeAnalysis.js`

Backend URL configuration:
- `VITE_API_BASE_URL`
- If unset, frontend uses same-origin path `/analyze`

## Local Setup

## 1) Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download NLTK stopwords once:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

Train model artifacts (first run):

```bash
python -m src.pipeline.training_pipeline
```

Build retrieval embeddings (recommended for current retriever path):

```bash
python scripts/build_embeddings.py
```

Set environment variables:

```bash
export GROQ_API_KEY='your_groq_key'
export NVIDIA_API_KEY='your_nvidia_key'   # optional but supported by judge node
```

Run API:

```bash
uvicorn main:app --reload --port 8000
```

## 2) Frontend

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Build frontend for production:

```bash
npm run build
npm run preview
```

## Data Requirements

Required raw files:
- `backend/data/raw/Fake.csv`
- `backend/data/raw/True.csv`

Without these files, training and embedding build scripts will fail.

## Deployment Notes

- Hosted frontend: https://news-credibility-monitor.vercel.app/
- Root `requirements_deploy.txt` provides a lightweight deployment dependency set.
- Frontend is a standard Vite static build output.
- Backend is a FastAPI service and can be containerized or deployed on any Python host.

## Known Implementation Notes

- The frontend sends `{ text, mode }` but backend currently reads only `text`.
- `POST /analyze` returns the final report object directly (not wrapped in an additional key).
- Retrieval path currently depends on precomputed embeddings load path used by the retriever helper.

## Report and Notebooks

- Technical report source: `Report/report.tex`
- Notebooks:
  - `notebooks/01_data_exploration.ipynb`
  - `notebooks/02_feature_engineering.ipynb`
  - `notebooks/03_model_comparison.ipynb`

## License

Academic project repository. See course/institution policies for reuse and distribution constraints.
