# Text-Event Anomaly Spike (IsolationForest + Rules)

Minimal spike on synthetic gameplay event logs (text only).
Goals: cheap scoring, simple features, operator-friendly reasons.

## Pipeline
1) Normalize events (lowercase, key=value tokens).
2) Vectorize with HashingVectorizer (no fit).
3) Train IsolationForest on “normal” windows.
4) Online scoring on sliding windows (+ rule checks).

## Run
python -m pip install -r requirements.txt
python main.py

Outputs top anomalies with reasons and window context.

## Notes
- This is a **toy** spike on synthetic logs to illustrate approach.
- Production: swap HashingVectorizer for engine-aware tokenization,
  add per-level baselines, persist model, and wire into C++ runtime
  (e.g., score via microservice or port rules + a compact scorer).
