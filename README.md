# Wisdom Volatility Scanner (v0.1)

A lightweight web app that analyzes weekly income volatility from a CSV file and returns
a quantitative risk score with optional AI-generated insights.

This project is an early prototype focused on correctness, clarity, and safe AI integration.

---

## What it does

- Accepts a CSV file with a required `amount` column
- Computes income volatility metrics:
  - Mean
  - Standard deviation
  - Coefficient of variation
  - Gap rate (near-zero income weeks)
  - Maximum drawdown
- Produces a volatility score (0–100) with a qualitative band
- Optionally generates AI explanations using OpenAI (gracefully degrades if unavailable)

---

## Tech stack

- Backend: FastAPI (Python)
- Frontend: Vanilla HTML + CSS + JavaScript
- AI: OpenAI API (optional)
- No database, no framework frontend

---

## Project structure
project/
├── backend/
│ └── main.py
├── frontend/
│ └── index.html
├── .env # API key (ignored by git)
├── .gitignore
└── README.md

---

## How to run locally

### Backend
```bash
python3 -m uvicorn backend.main:app --reload

http://127.0.0.1:8000

cd project
python3 -m http.server 5500

http://127.0.0.1:5500/frontend/index.html

date,amount
2024-01-01,500
2024-01-08,0
2024-01-15,800
