from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from io import StringIO
import os
import json
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def init_db():
    if not engine:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS analyses (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                n_weeks INTEGER,
                mean_weekly_income DOUBLE PRECISION,
                std_weekly_income DOUBLE PRECISION,
                coefficient_of_variation DOUBLE PRECISION,
                gap_rate DOUBLE PRECISION,
                max_drawdown DOUBLE PRECISION,
                volatility_score INTEGER,
                band TEXT
            );
        """))

@app.on_event("startup")
def _startup():
    init_db()

def save_analysis(result: dict):
    if not engine:
        return
    m = result["metrics"]
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO analyses
                (n_weeks, mean_weekly_income, std_weekly_income, coefficient_of_variation,
                 gap_rate, max_drawdown, volatility_score, band)
                VALUES
                (:n_weeks, :mean, :std, :cv, :gap, :dd, :score, :band)
            """),
            {
                "n_weeks": m["n_weeks"],
                "mean": m["mean_weekly_income"],
                "std": m["std_weekly_income"],
                "cv": m["coefficient_of_variation"],
                "gap": m["gap_rate"],
                "dd": m["max_drawdown"],
                "score": result["volatility_score"],
                "band": result["band"],
            }
        )


# OpenAI SDK (make sure installed: python3 -m pip install openai)
from openai import OpenAI

app = FastAPI(title="Wisdom Volatility Scanner")

# Allow local frontend dev; tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EPS = 1e-9

# Serve frontend (single-deploy setup)
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# ----------------------------
# Core analytics (deterministic)
# ----------------------------
def compute_max_drawdown_ratio(values: np.ndarray) -> float:
    """Max peak-to-trough drawdown ratio in [0, 1]."""
    peak = -np.inf
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / (peak + EPS) if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return float(max_dd)


def score_volatility(weekly_income: np.ndarray) -> dict:
    if len(weekly_income) < 4:
        raise ValueError("Need at least 4 weekly income values (8+ recommended).")

    x = np.array(weekly_income, dtype=float)
    if np.any(np.isnan(x)):
        raise ValueError("Income contains NaN values.")

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    cv = float(sigma / (mu + EPS)) if mu > 0 else 1.0  # if mean is 0, treat as highly unstable

    near_zero = max(25.0, 0.1 * mu)  # dynamic + floor
    gap_rate = float(np.mean(x < near_zero))

    max_drawdown = compute_max_drawdown_ratio(x)

    # Risks (0..1)
    cv_risk = min(cv / 0.6, 1.0)
    gap_risk = min(gap_rate / 0.25, 1.0)
    dd_risk = min(max_drawdown / 0.5, 1.0)

    risk = 0.5 * cv_risk + 0.3 * gap_risk + 0.2 * dd_risk
    volatility_score = int(round(100.0 * risk))

    # Optional “bands” for UX
    if volatility_score <= 30:
        band = "low"
    elif volatility_score <= 60:
        band = "moderate"
    else:
        band = "high"

    return {
        "metrics": {
            "n_weeks": int(len(x)),
            "mean_weekly_income": round(mu, 2),
            "std_weekly_income": round(sigma, 2),
            "coefficient_of_variation": round(cv, 3),
            "near_zero_threshold": round(near_zero, 2),
            "gap_rate": round(gap_rate, 3),
            "max_drawdown": round(max_drawdown, 3),
        },
        "components": {
            "cv_risk": round(cv_risk, 3),
            "gap_risk": round(gap_risk, 3),
            "drawdown_risk": round(dd_risk, 3),
            "weighted_risk": round(risk, 3),
        },
        "volatility_score": volatility_score,
        "band": band,
    }


def parse_csv(contents: str) -> np.ndarray:
    """
    Expected CSV columns:
      - amount (required)
    Optional:
      - date or week_start (ignored for scoring)
    """
    try:
        df = pd.read_csv(StringIO(contents))
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")

    cols = {c.lower().strip(): c for c in df.columns}
    if "amount" not in cols:
        raise ValueError("CSV must include an 'amount' column (weekly income).")

    amt_col = cols["amount"]
    amounts = pd.to_numeric(df[amt_col], errors="coerce").dropna().values

    if len(amounts) == 0:
        raise ValueError("No valid numeric amounts found in 'amount' column.")

    return amounts


# ----------------------------
# LLM explanation layer (non-deterministic, optional)
# ----------------------------
def generate_explanation(llm_payload: dict) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "signals": ["LLM explanation unavailable"],
            "explanation": "OPENAI_API_KEY is not set in the server environment.",
            "safe_next_steps": ["Set OPENAI_API_KEY and restart the server."]
        }

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are Wisdom, a financial MRI for gig workers. "
        "Explain computed income volatility metrics clearly and cautiously. "
        "Do NOT invent numbers. Do NOT give credit, legal, or medical advice. "
        "Use non-judgmental language."
    )

    user_prompt = (
        "Here are computed metrics and a volatility score:\n"
        f"{json.dumps(llm_payload, indent=2)}\n\n"
        "Return ONLY valid JSON. Do not include markdown, code fences, or any text outside the JSON object. "
        "The JSON must start with { and end with }. "
        "Keys required: signals (array of strings), explanation (string), safe_next_steps (array of strings)."

        "- signals: array of 1–3 short phrases\n"
        "- explanation: <= 120 words\n"
        "- safe_next_steps: array of 1–3 neutral suggestions\n"
        "Only explain what is supported by the metrics."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        # Keep the app functioning even if the LLM fails
        return {
            "signals": ["LLM request failed"],
            "explanation": f"Could not generate explanation due to an API error: {str(e)}",
            "safe_next_steps": ["Try again later or verify your API key/billing."]
        }

    text = resp.choices[0].message.content.strip()

    # Some models may wrap JSON in ```...```
    text = text.strip().strip("`")

    try:
        return json.loads(text)
    except Exception:
        return {
            "signals": ["Parsing error"],
            "explanation": text[:800],
            "safe_next_steps": ["Tighten the prompt or enforce JSON output more strictly."]
        }


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()

    try:
        contents = raw.decode("utf-8", errors="replace")
        weekly_income = parse_csv(contents)
        result = score_volatility(weekly_income)

        save_analysis(result)

        llm_payload = {
            "metrics": result["metrics"],
            "score": result["volatility_score"],
            "band": result["band"],
            "notes": "Explain the score using the metrics; do not invent numbers; be non-judgmental."
        }

        explanation = generate_explanation(llm_payload)

        return {
            "result": result,
            "explanation": explanation
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


