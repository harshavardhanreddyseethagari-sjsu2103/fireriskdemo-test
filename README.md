# Fire Risk Prediction — End-to-End MLOps Demo

A small ML system that predicts wildfire risk (Low / Medium / High) from
weather inputs, built to demonstrate the full path from a trained model
to a deployed, monitored, public service — not just a notebook.

**Live demo:** https://fireriskdemo-test.onrender.com/

## What this project covers

| Layer | What's here |
|---|---|
| Model | RandomForestClassifier trained on synthetic weather to fire-risk data (`generate_data.py`, `train_model.py`) |
| Serving | FastAPI app exposing `/predict` and `/health`, with Pydantic input validation (`app.py`) |
| Frontend | A dashboard UI calling the API directly (`static/index.html`) |
| Containerization | Docker image bundling the API, model, and dependencies (`Dockerfile`) |
| Monitoring | Structured JSON-lines logging of every request — successes, validation errors, and server errors — plus a summary script (`view_logs.py`) |
| Orchestration | A Prefect flow automating retraining, with a configurable on/off schedule (`pipeline_flow.py`, `deploy_pipeline.py`) |
| Cloud deployment | Deployed on Render, built directly from this repo's Dockerfile |

## Architecture note

Training/orchestration (Prefect) and serving (the deployed API) are
separate concerns, run independently. The orchestration layer
produces a `fire_risk_model.joblib` file; the serving layer loads
whatever model file currently exists. In a larger production setup,
these would be two separate deployments, with the training side
periodically pushing new model versions for the serving side to pick up.

## Running locally

```bash
# Train the model (one-time, or whenever you want to retrain)
python3 generate_data.py
python3 train_model.py

# Run the API directly (no Docker)
uvicorn app:app --reload
# visit http://127.0.0.1:8000/ for the dashboard
# visit http://127.0.0.1:8000/docs for the raw API

# View request logs after testing
python3 view_logs.py
```

## Running with Docker

```bash
docker build -t fire-risk-api .
docker run -p 8000:8000 -v $(pwd)/logs:/app/logs fire-risk-api
```

## Running the orchestration pipeline

```bash
# Run the training pipeline once, manually
python3 pipeline_flow.py

# Register a deployment (manual trigger or scheduled, see SCHEDULING_ENABLED)
python3 deploy_pipeline.py
```

## Known limitations

- Synthetic data, not real weather/fire data — this project is about the
  surrounding system, not the model itself.
- Model accuracy is moderate (81%); High-risk recall specifically is the
  metric that matters most for a safety use case and is weaker (67%) —
  noted but not tuned, since tuning wasn't the goal here.
- Orchestration (Prefect) is demonstrated locally but not wired into the
  live cloud deployment — see Architecture note above.