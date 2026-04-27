<div align="center">

# News Credibility Monitor

**An AI-powered news verification system combining ML classification, retrieval-augmented generation, and multi-agent LLM reasoning.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit_App-00C853?style=for-the-badge&logo=vercel&logoColor=white)](https://news-credibility-monitor.vercel.app/)
[![API](https://img.shields.io/badge/API-Render-4351e8?style=for-the-badge&logo=render&logoColor=white)](https://news-credibility-monitor-api.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)]()
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Pipeline-1C3C3C?style=flat-square&logo=langchain&logoColor=white)]()

</div>

---

| | |
|---|---|
| **Frontend** | https://news-credibility-monitor.vercel.app/ |
| **Backend API** | https://news-credibility-monitor-api.onrender.com/ |
| **ML Accuracy** | 98.88 % (ISOT dataset) |
| **Dataset** | ISOT Fake / True News (44,898 articles) |

---

## What It Does

Paste a news article → the system runs a **6-stage agentic pipeline** and returns a credibility verdict with full transparency: per-agent reasoning, evidence previews, agreement levels, and risk factors.

```
Article → Preprocessing → ML Classification → Conditional RAG → 3 LLM Agents → Judge → Verdict
```

## Key Features

- **Hybrid ML + LLM pipeline** — TF-IDF + Logistic Regression baseline feeds into a multi-agent LLM layer
- **Conditional RAG** — retrieval triggers only when ML confidence < 85%, saving latency on clear-cut cases
- **Multi-agent debate** — three agents (Conservative, Skeptical, Neutral) reason independently over the same evidence
- **Judge aggregation** — a separate judge model synthesizes a final verdict with conflict resolution
- **Risk transparency** — automated rule-based risk factors (low confidence, mixed evidence, weak consensus) surfaced in the UI
- **Animated pipeline visualization** — real-time step-by-step progress as the backend processes

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React 19, Vite 8, Tailwind CSS 4 | UI with animated pipeline visualization |
| Backend | FastAPI, Uvicorn | REST API (`POST /analyze`) |
| ML Model | scikit-learn (Logistic Regression) | TF-IDF baseline (98.88% accuracy) |
| Vector Store | ChromaDB (Ephemeral Client) | Conditional RAG retrieval |
| Embeddings | all-MiniLM-L6-v2 (precomputed) | 5,000 article embeddings |
| Agent Orchestration | LangGraph StateGraph | 8-node pipeline with conditional edges |
| LLM (Agents) | Llama 3.3-70B-Versatile | Groq API (temp 0.3) |
| LLM (Judge) | Llama 3.3-70B-Instruct | NVIDIA AI Endpoints (temp 0.2) |
| Deployment | Vercel + Render (free tier) | Frontend + backend hosting |
| Keep-Alive | GitHub Actions | Cron job every 14 min to prevent cold starts |

---

## Pipeline Architecture

```
                    ┌──────────────┐
                    │  User Input  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  preprocess  │  Strip datelines, lowercase, remove stopwords
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   ml_node    │  TF-IDF → Logistic Regression → verdict + confidence
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Confidence  │
                    │   ≥ 85% ?    │
                    └──┬───────┬───┘
                  Yes  │       │  No
                       │  ┌────▼────┐
                       │  │rag_node │  ChromaDB top-5 retrieval
                       │  └────┬────┘
                       │       │
                    ┌──▼───────▼───┐
                    │   Agent A    │  Conservative — trusts ML & data
                    │   Agent B    │  Skeptical — challenges evidence
                    │   Agent C    │  Neutral — balanced weighing
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Judge Node  │  Synthesizes final verdict + consensus
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Output Node  │  Risk factors, agreement, structured report
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Response   │
                    └──────────────┘
```

---

## Repository Structure

```
News-Credibility-Monitor/
├── backend/
│   ├── main.py                          # FastAPI entrypoint
│   ├── requirements.txt                 # Full dependencies
│   ├── requirements_deploy.txt          # Lean deployment deps (no PyTorch)
│   ├── render.yaml                      # Render service config
│   ├── data/raw/
│   │   ├── Fake.csv                     # ISOT Fake News
│   │   └── True.csv                     # ISOT True News
│   ├── models/
│   │   ├── best_model.pkl               # Trained Logistic Regression
│   │   ├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│   │   └── embeddings.pkl               # Precomputed MiniLM embeddings
│   ├── scripts/
│   │   └── build_embeddings.py          # Embedding generation script
│   └── src/
│       ├── agent/
│       │   ├── graph.py                 # LangGraph StateGraph definition
│       │   ├── nodes.py                 # 8 node functions + routing logic
│       │   └── state.py                 # AgentState TypedDict
│       ├── llm/
│       │   ├── client.py                # Groq API client with retry logic
│       │   └── prompts.py               # Agent & judge prompt templates
│       ├── rag/
│       │   ├── load_embeddings.py       # ChromaDB ephemeral loader
│       │   └── retriever.py             # Semantic similarity search (k=5)
│       ├── models/
│       │   ├── train.py                 # LogReg training
│       │   └── evaluate.py              # Metrics & confusion matrix
│       ├── features/
│       │   └── build_features.py        # TF-IDF vectorizer builder
│       ├── data/
│       │   └── load_data.py             # CSV loading & merging
│       ├── pipeline/
│       │   └── training_pipeline.py     # End-to-end training orchestration
│       ├── config/
│       │   └── config.py                # Paths & constants
│       └── utils/
│           └── text_cleaner.py          # Regex cleaning + stopword removal
├── frontend/
│   ├── src/
│   │   ├── App.jsx                      # Main app with pipeline animation
│   │   ├── lib/
│   │   │   └── normalizeAnalysis.js     # Backend → UI payload transform
│   │   └── components/
│   │       ├── InputSection.jsx         # Article input + word counter
│   │       ├── VerdictCard.jsx          # Final verdict display
│   │       ├── AgentSection.jsx         # 3 agent cards (expandable)
│   │       ├── AgreementSection.jsx     # Vote distribution
│   │       ├── EvidenceSection.jsx      # RAG doc previews
│   │       ├── RiskSection.jsx          # Risk factor badges
│   │       ├── ThinkingPipeline.jsx     # Animated step progress
│   │       └── background/
│   │           └── SoftAuroraBackground.jsx  # WebGL shader background
│   ├── vercel.json                      # Vercel SPA config
│   └── package.json
├── .github/workflows/
│   └── keep-alive.yml                   # Cron ping every 14 min
├── Report/
│   └── report.tex                       # LaTeX project report
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
└── docs/
```

---

## API Reference

### `GET /`

Health check.

```json
{ "status": "ok", "message": "News Credibility Monitor API is running" }
```

### `POST /analyze`

Analyze an article for credibility.

**Request:**
```json
{
  "text": "Article text (minimum 50 words)...",
  "mode": "agentic"
}
```

**Response:**
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
  "risk_factors": ["Low ML confidence (<70%)", "Weak consensus (2 vs 1 split)"],
  "ml_signal": "REAL (91.2%)",
  "rag_count": 5,
  "error": null
}
```

---

## Local Development

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

**First-time setup** — train ML model and build embeddings:
```bash
python -m src.pipeline.training_pipeline
python scripts/build_embeddings.py
```

**Set API keys and run:**
```bash
export GROQ_API_KEY='your_groq_key'
export NVIDIA_API_KEY='your_nvidia_key'
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

**Production build:**
```bash
npm run build && npm run preview
```

---

## Deployment

| Service | Platform | Config |
|---|---|---|
| Frontend | Vercel | `frontend/vercel.json` — SPA rewrite to `index.html` |
| Backend | Render (free tier) | `render.yaml` — uvicorn, 1 worker, 512 MB |
| Keep-Alive | GitHub Actions | `.github/workflows/keep-alive.yml` — cron `*/14 * * * *` |

**Environment variables** (set in Render dashboard):
- `GROQ_API_KEY` — Groq API key for agent LLM calls
- `NVIDIA_API_KEY` — NVIDIA AI Endpoints key for judge model
- `CORS_ALLOW_ORIGINS` — allowed frontend origins

The deployment uses `requirements_deploy.txt` (excludes PyTorch, Jupyter, sentence-transformers) to fit within Render's 512 MB free-tier memory limit.

---

## ML Model Performance

Evaluated on a 20% held-out split of the ISOT Fake News Dataset.

| Metric | Score |
|---|---|
| Accuracy | 98.88% |
| Precision (weighted) | 98.88% |
| Recall (weighted) | 98.88% |
| F1 (weighted) | 98.88% |

**Model:** Logistic Regression (`class_weight="balanced"`, solver `lbfgs`, max_iter 1000)
**Features:** TF-IDF vectorizer (10,000 features, unigram + bigram)

---

## Data Requirements

Required raw files (not committed to repo):
- `backend/data/raw/Fake.csv`
- `backend/data/raw/True.csv`

Source: [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php)

---

## Report & Notebooks

| Document | Path |
|---|---|
| LaTeX Report | `Report/report.tex` |
| Data Exploration | `notebooks/01_data_exploration.ipynb` |
| Feature Engineering | `notebooks/02_feature_engineering.ipynb` |
| Model Comparison | `notebooks/03_model_comparison.ipynb` |

---

## License

Academic project repository. See course/institution policies for reuse and distribution constraints.
# News_Credibility
