-- Wisdom v0.2 (minimal schema)

-- 1) Users (keeps the door open for real accounts later)
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2) Raw weekly income entries (this is your ground truth dataset)
CREATE TABLE IF NOT EXISTS weekly_income_entries (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  week_start DATE NOT NULL,            -- e.g., 2026-01-05
  amount NUMERIC(12,2) NOT NULL,        -- money, exact
  source TEXT NOT NULL DEFAULT 'manual',-- manual|csv|bank|etc
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- prevents duplicates for the same week
  UNIQUE(user_id, week_start)
);

-- 3) Analysis results (what your API computes + AI explanation)
CREATE TABLE IF NOT EXISTS analysis_results (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  window_start DATE NOT NULL,
  window_end DATE NOT NULL,

  volatility_score INT NOT NULL,
  band TEXT NOT NULL, -- low/moderate/high

  metrics_json JSONB NOT NULL,          -- mean, std, cv, gap, drawdown, etc
  explanation_json JSONB,               -- signals/explanation/next steps (nullable)

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_income_user_week ON weekly_income_entries(user_id, week_start);
CREATE INDEX IF NOT EXISTS idx_analysis_user_created ON analysis_results(user_id, created_at DESC);
