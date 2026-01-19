from __future__ import annotations

import os
import json
import random
from pathlib import Path
from io import StringIO
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# --- Optional OpenAI import (don’t crash if missing locally) ---
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

load_dotenv()

EPS = 1e-9

app = FastAPI(title="Wisdom • Volatility + Spend Benchmark")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Frontend serving (robust)
# ----------------------------
# If file is backend/main.py:
#   project_root = backend/.. = project root
#   frontend expected at project_root/frontend/index.html
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Fallback: if no frontend folder, serve from backend dir
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BACKEND_DIR

# Serve assets if you add any later
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"index.html not found at {index_path}. Put your index.html in {FRONTEND_DIR}/",
        )
    return FileResponse(str(index_path))


@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Database (Postgres on Render)
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

def init_volatility_db():
    """Create volatility table if DB configured."""
    if not engine:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    user_id TEXT,
                    n_weeks INTEGER,
                    mean_weekly_income DOUBLE PRECISION,
                    std_weekly_income DOUBLE PRECISION,
                    coefficient_of_variation DOUBLE PRECISION,
                    gap_rate DOUBLE PRECISION,
                    max_drawdown DOUBLE PRECISION,
                    volatility_score INTEGER,
                    band TEXT
                );
                """
            )
        )


def init_spend_db():
    """Create spend benchmark tables if DB configured."""
    if not engine:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS spend_submissions (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    user_id TEXT,
                    frequency TEXT NOT NULL CHECK (frequency IN ('daily','weekly','monthly')),
                    avg_spend_pct DOUBLE PRECISION NOT NULL,
                    std_spend_pct DOUBLE PRECISION NOT NULL
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS spend_entries (
                    id SERIAL PRIMARY KEY,
                    submission_id INTEGER NOT NULL REFERENCES spend_submissions(id) ON DELETE CASCADE,
                    idx SMALLINT NOT NULL CHECK (idx >= 0 AND idx <= 6),
                    spend_pct DOUBLE PRECISION NOT NULL CHECK (spend_pct >= 0 AND spend_pct <= 100)
                );
                """
            )
        )


def seed_spend_demo_data_if_empty(n_per_freq: int = 100):
    """
    Seed demo crowd data so percentile works immediately.
    Only runs if spend_submissions is empty.
    """
    if not engine:
        return

    with engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM spend_submissions")).scalar_one()
        if int(total or 0) > 0:
            return  # already seeded or real users exist

        for freq in ("daily", "weekly", "monthly"):
            # slightly different spread by cadence
            base_std = 6 if freq == "daily" else (8 if freq == "weekly" else 10)

            for i in range(n_per_freq):
                # avg centered ~65, spread ~15
                avg_target = max(5.0, min(98.0, random.gauss(65, 15)))
                vals = [max(0.0, min(100.0, random.gauss(avg_target, base_std))) for _ in range(7)]

                avg = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                res = conn.execute(
                    text(
                        """
                        INSERT INTO spend_submissions (user_id, frequency, avg_spend_pct, std_spend_pct)
                        VALUES (:user_id, :frequency, :avg, :std)
                        RETURNING id;
                        """
                    ),
                    {
                        "user_id": f"demo_{freq}_{i}",
                        "frequency": freq,
                        "avg": avg,
                        "std": std,
                    },
                )
                submission_id = int(res.scalar_one())

                rows = [{"submission_id": submission_id, "idx": j, "spend_pct": float(vals[j])} for j in range(7)]
                conn.execute(
                    text(
                        """
                        INSERT INTO spend_entries (submission_id, idx, spend_pct)
                        VALUES (:submission_id, :idx, :spend_pct);
                        """
                    ),
                    rows,
                )


@app.on_event("startup")
def on_startup():
    try:
        init_volatility_db()
        init_spend_db()
        seed_spend_demo_data_if_empty(100)
    except Exception as e:
        print(f"[WARN] Startup init failed: {e}")


# ----------------------------
# Volatility analytics (existing)
# ----------------------------
def compute_max_drawdown_ratio(values: np.ndarray) -> float:
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
    cv = float(sigma / (mu + EPS)) if mu > 0 else 1.0

    near_zero = max(25.0, 0.1 * mu)
    gap_rate = float(np.mean(x < near_zero))

    max_drawdown = compute_max_drawdown_ratio(x)

    cv_risk = min(cv / 0.6, 1.0)
    gap_risk = min(gap_rate / 0.25, 1.0)
    dd_risk = min(max_drawdown / 0.5, 1.0)

    risk = 0.5 * cv_risk + 0.3 * gap_risk + 0.2 * dd_risk
    volatility_score = int(round(100.0 * risk))

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


def save_volatility_analysis(result: dict, user_id: Optional[str] = None):
    if not engine:
        return
    m = result["metrics"]
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO analyses
                    (user_id, n_weeks, mean_weekly_income, std_weekly_income, coefficient_of_variation,
                     gap_rate, max_drawdown, volatility_score, band)
                    VALUES
                    (:user_id, :n_weeks, :mean, :std, :cv, :gap, :dd, :score, :band)
                    """
                ),
                {
                    "user_id": user_id,
                    "n_weeks": m["n_weeks"],
                    "mean": m["mean_weekly_income"],
                    "std": m["std_weekly_income"],
                    "cv": m["coefficient_of_variation"],
                    "gap": m["gap_rate"],
                    "dd": m["max_drawdown"],
                    "score": result["volatility_score"],
                    "band": result["band"],
                },
            )
    except Exception as e:
        print(f"[WARN] save_volatility_analysis failed: {e}")


def generate_explanation(llm_payload: dict) -> dict:
    # graceful fallback if OpenAI not configured
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return {
            "signals": ["LLM explanation unavailable"],
            "explanation": "OPENAI_API_KEY is not set (or OpenAI SDK not installed).",
            "safe_next_steps": ["Set OPENAI_API_KEY to enable AI narration."],
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
        "Return ONLY valid JSON (no markdown). Keys: signals (array), explanation (string), safe_next_steps (array). "
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
        out = (resp.choices[0].message.content or "").strip().strip("`")
        return json.loads(out)
    except Exception as e:
        return {
            "signals": ["LLM request failed"],
            "explanation": f"Could not generate explanation due to an error: {str(e)}",
            "safe_next_steps": ["Try again later or verify API key."],
        }


class AnalyzeJSONRequest(BaseModel):
    user_id: Optional[str] = None
    amounts: List[float]


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), user_id: str = Form(None)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()
    try:
        contents = raw.decode("utf-8", errors="replace")
        weekly_income = parse_csv(contents)
        result = score_volatility(weekly_income)

        save_volatility_analysis(result, user_id=user_id)

        llm_payload = {
            "metrics": result["metrics"],
            "score": result["volatility_score"],
            "band": result["band"],
        }
        explanation = generate_explanation(llm_payload)
        return {"result": result, "explanation": explanation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze-json")
def analyze_json(payload: AnalyzeJSONRequest):
    try:
        weekly_income = np.array(payload.amounts, dtype=float)
        if len(weekly_income) < 4:
            raise ValueError("Please provide at least 4 weekly income values.")

        result = score_volatility(weekly_income)
        save_volatility_analysis(result, user_id=payload.user_id)

        llm_payload = {
            "metrics": result["metrics"],
            "score": result["volatility_score"],
            "band": result["band"],
        }
        explanation = generate_explanation(llm_payload)
        return {"result": result, "explanation": explanation}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ----------------------------
# Spend benchmark (NEW)
# ----------------------------
class SpendAnalyzeRequest(BaseModel):
    user_id: Optional[str] = None
    frequency: str  # daily|weekly|monthly
    values: List[float]  # length 7, each 0..100


def _validate_frequency(freq: str) -> str:
    f = (freq or "").strip().lower()
    if f not in ("daily", "weekly", "monthly"):
        raise ValueError("frequency must be one of: daily, weekly, monthly")
    return f


def _validate_values(values: List[float]) -> np.ndarray:
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError("values must be an array of exactly 7 numbers")
    x = np.array(values, dtype=float)
    if np.any(np.isnan(x)):
        raise ValueError("values contains NaN")
    if np.any(x < 0) or np.any(x > 100):
        raise ValueError("each value must be between 0 and 100")
    return x


def save_spend_submission(user_id: Optional[str], frequency: str, values: np.ndarray) -> Optional[int]:
    if not engine:
        return None

    avg = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO spend_submissions (user_id, frequency, avg_spend_pct, std_spend_pct)
                VALUES (:user_id, :frequency, :avg, :std)
                RETURNING id;
                """
            ),
            {"user_id": user_id, "frequency": frequency, "avg": avg, "std": std},
        )
        submission_id = int(res.scalar_one())

        rows = [{"submission_id": submission_id, "idx": i, "spend_pct": float(values[i])} for i in range(7)]
        conn.execute(
            text(
                """
                INSERT INTO spend_entries (submission_id, idx, spend_pct)
                VALUES (:submission_id, :idx, :spend_pct);
                """
            ),
            rows,
        )
    return submission_id


def compute_percentile_rank(frequency: str, user_avg: float) -> Optional[float]:
    """
    Percentile rank among all submissions of same frequency.
    Returns 0..100, or None if no DB / no data.
    """
    if not engine:
        return None

    with engine.begin() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM spend_submissions WHERE frequency = :f"),
            {"f": frequency},
        ).scalar_one()
        total = int(total or 0)
        if total == 0:
            return None

        below = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM spend_submissions
                WHERE frequency = :f AND avg_spend_pct < :a
                """
            ),
            {"f": frequency, "a": float(user_avg)},
        ).scalar_one()
        below = int(below or 0)

    pct = 100.0 * (below / total)
    return round(pct, 1)

class SpendBenchmarkRequest(BaseModel):
    frequency: str  # daily|weekly|monthly


def get_spend_benchmark(frequency: str, bins: int = 20) -> dict:
    """
    Returns histogram + median for avg_spend_pct among all users for a frequency.
    """
    if not engine:
        edges = np.linspace(0, 100, bins + 1)
        counts = [random.randint(2, 22) for _ in range(bins)]
        return {
            "median": 65.0,
            "bin_edges": [round(float(x), 2) for x in edges.tolist()],
            "counts": counts,
            "n": sum(counts),
        }

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT avg_spend_pct
                FROM spend_submissions
                WHERE frequency = :f
            """),
            {"f": frequency},
        ).fetchall()

    values = [float(r[0]) for r in rows if r[0] is not None]
    n = len(values)
    if n == 0:
        edges = np.linspace(0, 100, 21)  # 20 bins => 21 edges
        return {
            "median": None,
            "bin_edges": [round(float(x), 2) for x in edges.tolist()],
            "counts": [0] * 20,
            "n": 0,
        }

    values_np = np.array(values, dtype=float)
    median = float(np.median(values_np))

    # bins between 0..100 (percent)
    edges = np.linspace(0, 100, bins + 1)
    counts, _ = np.histogram(values_np, bins=edges)

    # Return as JSON-friendly lists
    return {
        "median": round(median, 2),
        "bin_edges": [round(float(x), 2) for x in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "n": n,
    }


@app.post("/spend/benchmark")
def spend_benchmark(payload: SpendBenchmarkRequest):
    try:
        f = _validate_frequency(payload.frequency)
        bench = get_spend_benchmark(f, bins=20)
        return {"result": {"frequency": f, **bench}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] /spend/benchmark failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error")


@app.post("/spend/analyze")
def spend_analyze(payload: SpendAnalyzeRequest):
    try:
        frequency = _validate_frequency(payload.frequency)
        values = _validate_values(payload.values)

        avg = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        submission_id = save_spend_submission(payload.user_id, frequency, values)
        percentile = compute_percentile_rank(frequency, avg)

        return {
            "result": {
                "submission_id": submission_id,
                "frequency": frequency,
                "values": [float(v) for v in values],
                "avg_spend_pct": round(avg, 2),
                "std_spend_pct": round(std, 2),
                "percentile_rank": percentile,  # should exist immediately if DB + seed worked
                "note": "Percentile is relative to users of this tool, not the general population.",
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] /spend/analyze failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error")


# ----------------------------
# Local run convenience
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
