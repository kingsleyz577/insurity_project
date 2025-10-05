# insurity_project
Telematics Integration in Auto Insurance — Proof of Concept (PoC)
================================================================

Repo: https://github.com/<your-username>/telematics-insurance-poc
Contact: <name> (<email>)

──────────────────────────────────────────────────────────────────
0) TL;DR Demo
──────────────────────────────────────────────────────────────────
# 1) Create env + install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Generate synthetic trips + labels
python bin/simulate.py --drivers 500 --days 30 --out data/raw/

# 3) Build features (trip/day/driver aggregates)
python bin/build_features.py --in data/raw --out data/feature_store

# 4) Train (frequency + severity + pricing)
python bin/train.py --features data/feature_store --models models/

# 5) Serve APIs (ingest, score, price)
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 6) (Optional) Launch dashboard
streamlit run src/dashboard.py

# 7) Quick eval report
python bin/eval.py --features data/feature_store --models models/

──────────────────────────────────────────────────────────────────
1) Background & Objective
──────────────────────────────────────────────────────────────────
Traditional pricing leans on demographics/history; it under-represents actual behavior.
Telematics (GPS + accelerometer + trip context) enables usage-based insurance (PAYD/PHYD).
Goal: capture real driving, score risk, and connect score → dynamic premiums with transparency.

Targets
- Improve premium accuracy (lower loss ratio volatility; better calibration).
- Encourage safer driving (incentives, feedback, gamification).
- Customer transparency (scores, reasons, “what to improve”).
- Data security & privacy by design (consent, minimization, encryption, retention).

──────────────────────────────────────────────────────────────────
2) System Architecture (PoC)
──────────────────────────────────────────────────────────────────
Data Sources
- Telematics events (simulated): timestamp, driver_id, speed, accel/brake, heading, gps (lat/lon),
  trip_id, phone_motion (optional).
- Contextual: weather (optional stub), road/area risk index (simulated), crime index (simulated).

Pipeline (near-real-time capable; batch in PoC)
- Ingest API (FastAPI): POST /v1/ingest/events → append-only store (Parquet under data/raw/).
- Feature Engineering (batch): bin/build_features.py aggregates events into trip/day/driver features
  (e.g., harsh_brakes_per_100mi, night_miles_pct, speeding_time_pct, route_risk_avg).
- Modeling:
  • Frequency (claim yes/no): Gradient Boosting (LightGBM-style via xgboost/sklearn) / Logistic GLM baseline.
  • Severity (claim cost): Tweedie/Gamma regression baseline + Gradient Boosting regressor.
  • Expected Loss = P(claim)*E(cost|claim). Calibrate with isotonic/Platt as needed.
- Pricing:
  • Premium = (Expected Loss × Safety Loading) + Fixed Expense + Risk Margin.
  • Capping/guards & monotonicity checks on key features.
- Serving:
  • /v1/score (features → risk_score, expected_loss)
  • /v1/price (risk_score → quoted premium, with breakdown)
  • /v1/explain (top SHAP-like reasons; PoC uses model-agnostic permutation importances)
- Dashboard (Streamlit):
  • Driver view: weekly score trend, premium deltas, “how to improve”.
  • Portfolio view: calibration plots, lift charts, loss ratio vs deciles.

Security & Privacy (PoC stance)
- Separate PII from telemetry (logical “PII vault” in data/pii/).
- Encrypt-at-rest (optionally via filesystem tools) & TLS in prod; PoC focuses on structure.
- Consent flags per driver_id; sampling/minimization; retention config (bin/retention.py).
- Location accuracy throttling + k-anonymity style coarse geohash aggregation for analytics.

──────────────────────────────────────────────────────────────────
3) Data Model (ER Outline)
──────────────────────────────────────────────────────────────────
Tables (stored as Parquet/CSV in PoC; RDBMS-ready schemas in docs/ddl.sql)

drivers( pk driver_id, ak policy_number, consent_flag, created_at )
policies( pk policy_id, ak policy_number, driver_id→drivers, effective_date, status )
trips( pk trip_id, ppk driver_id→drivers, start_ts, end_ts, miles, night_flag )
events( pk event_id, ppk trip_id→trips, ts, lat, lon, speed_mps, accel_mps2, brake_flag, heading_deg )
daily_features( pk driver_id + date, harsh_brakes_p100mi, speeding_pct, night_miles_pct, route_risk_avg, weather_risk_avg, mileage )
trip_features( pk trip_id, mean_speed, max_decel, cornering_events, stop_go_index, congestion_pct )
labels( pk driver_id + date, claim_flag, claim_cost )   # simulated truth for training
pricing_quotes( pk quote_id, ppk driver_id, ts, risk_score, expected_loss, premium_components_json )

Notation follows your preference: pk/ppk/ak/pak.

Sample telematics event JSON (POST /v1/ingest/events)
{
  "driver_id": "D_00123",
  "trip_id": "T_abc123",
  "events": [
    {"ts":"2025-06-01T14:03:12Z","lat":42.36,"lon":-71.06,"speed_mps":16.4,"accel_mps2":-3.2,"brake_flag":1,"heading_deg":120},
    ...
  ]
}

──────────────────────────────────────────────────────────────────
4) Features (examples)
──────────────────────────────────────────────────────────────────
Driving Behavior
- Harsh brakes per 100 miles, hard acceleration per 100 miles
- Speeding time % (> posted limit; PoC uses synthetic “limit” by area type)
- Night miles %, adverse weather miles %, stop-and-go index
- Cornering (lateral acceleration proxy), phone_motion events (if available)
Usage/Exposure
- Daily miles, trip count, average trip length, commute hour ratio
Contextual
- Area risk index (sim), crime index (sim), road class mix (highway/arterial/local), weather risk

──────────────────────────────────────────────────────────────────
5) Modeling Approach
──────────────────────────────────────────────────────────────────
Targets
- Frequency: claim_flag ∈ {0,1} (driver-day basis).
- Severity: claim_cost ≥ 0 (driver-day with claim_flag=1).
- Combined Expected Loss (EL): P(claim) × E(cost|claim).

Models
- Baselines: LogisticRegression (freq), TweedieRegressor with power∈[1,2] (sev).
- Stronger: Gradient Boosting (XGBoost/LightGBM equivalent via xgboost).
- Calibration: isotonic for freq; simple scaling for sev.
- Validation: time-based split (last 20% days holdout).
- Metrics:
  • Frequency: AUROC, AUPRC, Brier, Calibration (ECE), log loss.
  • Severity: MAE, RMSE on log(1+cost), Tweedie deviance.
  • Portfolio: Lift (top decile), Gini, LR vs deciles.

Risk Score
- score = percentile_rank(EL within portfolio); report [0,100].
- Display top contributing features per driver-day.

Pricing (illustrative, not actuarial advice)
- Premium = EL × (1 + safety_loading) + expense_load + risk_margin.
- Guards: min/max premium caps, change limits (e.g., ±10%/month), fairness checks.

──────────────────────────────────────────────────────────────────
6) API (FastAPI)
──────────────────────────────────────────────────────────────────
GET  /healthz -> {"ok":true}
POST /v1/ingest/events -> {accepted: N}
POST /v1/score       (driver_id|features) -> {risk_score, expected_loss, explanations}
POST /v1/price       (driver_id|risk_score) -> {premium_breakdown, final_premium}
GET  /v1/driver/<id>/history -> time series of scores/premiums

Example: curl -X POST localhost:8000/v1/price -H "Content-Type: application/json" -d '
{"driver_id":"D_00123","risk_score":72}
'

──────────────────────────────────────────────────────────────────
7) Dashboard (Streamlit quickstart)
──────────────────────────────────────────────────────────────────
- Driver: weekly score, premium delta, top 3 focus areas (e.g., "reduce night miles", "fewer hard brakes").
- Portfolio: calibration plot, lift chart, segment drilldowns.

Run: streamlit run src/dashboard.py

──────────────────────────────────────────────────────────────────
8) Exact Setup, Run & Eval (detailed)
──────────────────────────────────────────────────────────────────
(1) Environment
    Python 3.10+
    pip install -r requirements.txt
    # Key libs: fastapi, uvicorn, pydantic, numpy, pandas, pyarrow, scikit-learn, xgboost, shap, streamlit

(2) Data Simulation
    python bin/simulate.py --drivers 500 --days 30 --out data/raw/
    # Produces:
    #   data/raw/events/ (event parquet shards)
    #   data/raw/labels/ (driver-day labels: claim_flag, claim_cost)
    #   data/pii/drivers.csv (minimal PII with consent_flag)

(3) Feature Build
    python bin/build_features.py --in data/raw --out data/feature_store
    # Produces trip_features.parquet, daily_features.parquet, training.parquet

(4) Train
    python bin/train.py --features data/feature_store --models models/
    # Saves:
    #   models/frequency.pkl, models/severity.pkl, models/calibration.pkl
    #   models/metadata.json (feature list, version hash)

(5) Serve
    uvicorn src.api:app --host 0.0.0.0 --port 8000

(6) Evaluate
    python bin/eval.py --features data/feature_store --models models/
    # Prints AUROC, AUPRC, Brier, MAE, lift, calibration; writes docs/reports/*.html

(7) Dashboard
    streamlit run src/dashboard.py

──────────────────────────────────────────────────────────────────
9) File / Dir Layout
──────────────────────────────────────────────────────────────────
/src
  api.py                # FastAPI app (ingest, score, price, explain)
  feature_builder.py    # event->trip/day/driver aggregations
  models.py             # wrappers for freq/sev models + calibration
  pricing.py            # premium calculator + guards
  explain.py            # simple permutation-importance explanations
  utils_io.py           # parquet/csv helpers, PII separation
  schema.py             # pydantic request/response schemas
  privacy.py            # consent checks, retention utilities

/bin
  simulate.py           # synthetic trips/events + labels
  build_features.py     # batch feature job
  train.py              # training CLI
  eval.py               # metrics + plots
  retention.py          # deletes/aggregates per retention policy

/models                 # saved model artifacts (git-ignored by default)
/data
  /raw                  # synthetic events + labels
  /feature_store        # engineered features
  /pii                  # minimal PII (separate tree)

/docs
  design.md             # architecture notes + diagrams (ASCII/mermaid)
  ddl.sql               # optional RDBMS DDL mirroring PoC tables
  pricing-notes.md      # formulae, caps, loadings
  security-privacy.md   # minimization, consent, retention, geohash strategy

/requirements.txt
/README (this file)
(.env.example)          # e.g., API keys if you add weather/traffic

──────────────────────────────────────────────────────────────────
10) Modeling Notes
──────────────────────────────────────────────────────────────────
- Baseline GLM provides transparency and establishes a fair yardstick.
- Gradient boosting improves lift; keep monotone constraints on speed/harshness features if needed.
- Calibrate frequency; sanity-check severity tails with capping/winsorization.
- Evaluate: prioritize calibration (pricing needs well-calibrated probabilities), not just ranking.

──────────────────────────────────────────────────────────────────
11) Security, Privacy, Compliance (PoC posture)
──────────────────────────────────────────────────────────────────
- Consent first: events processed only if drivers.consent_flag = true.
- Data minimization: drop raw GPS after aggregations for pricing; retain coarse geohash only for analytics.
- Retention: default 180 days for raw events; aggregations kept 24 months.
- Separation of concerns: PII stored separately from telemetry; join via driver_id only when necessary.
- In production: enforce TLS, KMS at-rest encryption, audit trails, access control (RBAC).

──────────────────────────────────────────────────────────────────
12) Nice-to-Have Hooks (stubs included)
──────────────────────────────────────────────────────────────────
- Gamification: badges + streaks in dashboard; reward rules in pricing-notes.md.
- Real-time in-trip feedback: placeholder WebSocket in api.py (commented).
- Weather/traffic: src/context_providers.py with mock providers; swap to real APIs later.
- Smart city data: extend route_risk_avg provider to accept external tiles.

──────────────────────────────────────────────────────────────────
13) Cost & ROI (PoC framing)
──────────────────────────────────────────────────────────────────
- PoC uses local files/Streamlit to keep infra costs ~zero.
- ROI levers: improved selection (lift), fraud reduction signals (phone_motion/night anomalies),
  retention via transparency/gamification; target is improved combined ratio over baseline.

──────────────────────────────────────────────────────────────────
14) Real-World Testing
──────────────────────────────────────────────────────────────────
- Shadow pricing: run PoC model alongside legacy rating for a holdout cohort; compare calibration & LR.
- A/B: capped monthly change (±10%) to manage customer fairness.
- Monitor: drift on key features, score stability, and premium changes distribution.

──────────────────────────────────────────────────────────────────
15) In the future, for further improvements
──────────────────────────────────────────────────────────────────
- Move features to a Feature Store (e.g., Feast) + streaming (Kafka/Flink/Spark).
- Monotonic GBMs or constrained GAMs for more interpretable, safer pricing.
- Differential privacy for analytics; on-device pre-aggregation to reduce raw GPS handling.
- Fairness audits across age/region/vehicle proxies; adversarial debiasing if needed.
- Add severity mixture models (zero-inflated Tweedie), quantile pricing for tail risk.

──────────────────────────────────────────────────────────────────
16) License & Disclaimer
──────────────────────────────────────────────────────────────────
- Educational PoC; not actuarial or legal advice. Do not deploy to production as-is.
- Synthetic data only; remove PII before sharing models.
