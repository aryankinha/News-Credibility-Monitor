import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env from the backend directory (sibling of this file).
load_dotenv()

# NOTE: heavy ML imports (sentence-transformers, chromadb, sklearn models)
# are deferred to first request so the port binds before the 512Mi free-tier
# limit is hit at boot. Do NOT import run_agent at module top.
_run_agent = None


def get_run_agent():
    global _run_agent
    if _run_agent is None:
        from src.agent.graph import run_agent  # noqa: WPS433 (lazy import on purpose)

        _run_agent = run_agent
    return _run_agent

app = FastAPI(title="News Credibility Monitor API")

# CORS — allow the deployed frontend (Vercel) plus local dev to hit the API.
# Set CORS_ALLOW_ORIGINS in the Render dashboard as a comma-separated list,
# e.g. "https://your-app.vercel.app,https://your-app-git-main.vercel.app".
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_env_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
allow_origins = _env_origins or _default_origins

# Optional: regex to match Vercel preview deployments
# (e.g. https://news-credibility-monitor-git-*-yourname.vercel.app).
allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "ok", "message": "News Credibility Monitor API is running"}


@app.post("/analyze")
def analyze(data: dict):
    text = data.get("text", "")

    if not text or len(text.split()) < 50:
        return {"error": "Text too short. Please provide at least 50 words for analysis."}

    try:
        # Lazy-load the agent graph so cold-boot fits in Render's 512Mi free tier.
        result = get_run_agent()(text)
        return result
    except Exception as e:
        return {"error": str(e)}
