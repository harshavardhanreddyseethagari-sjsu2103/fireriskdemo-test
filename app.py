# app.py
#
# Same as before, but now every request to /predict gets logged as one
# line of JSON in a file called "request_log.jsonl". Each line is a
# complete, independent JSON object — that's what the "jsonl" (JSON Lines)
# format means, as opposed to one giant JSON array. JSON Lines is the
# industry-standard format for logs because you can append to the file
# forever without ever needing to rewrite the whole thing.

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
import time
from datetime import datetime, timezone
import os

app = FastAPI(title="Fire Risk Prediction API")

# CORS = Cross-Origin Resource Sharing. Browsers block a webpage from
# calling an API on a DIFFERENT origin (domain/port) unless that API
# explicitly allows it. allow_origins=["*"] means "any origin can call
# this API" — fine for a portfolio demo, NOT something you'd do for a
# real product handling sensitive data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("fire_risk_model.joblib")

os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/request_log.jsonl"


# ── Serve the dashboard ────────────────────────────────────────
# Remember way back when you got a 404 visiting "/"? That was because
# no route existed for it. This fixes that — now "/" returns the
# dashboard HTML file directly, so a human in a browser sees a real UI
# instead of a bare JSON API with no front door. "/docs" still works
# exactly as before, for anyone who wants to test the raw API itself.
@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")


class WeatherInput(BaseModel):
    # ge = "greater than or equal", le = "less than or equal".
    # These ranges mirror the bounds used in generate_data.py — anything
    # outside what the model was ever trained on is rejected up front,
    # instead of letting the model silently guess on nonsense.
    temperature: float = Field(ge=-10, le=55, description="Celsius")
    humidity:    float = Field(ge=0,   le=100, description="Percent")
    wind_speed:  float = Field(ge=0,   le=150, description="km/h")
    rainfall:    float = Field(ge=0,   le=200, description="mm in last 24h")


def log_event(event: dict):
    """
    Appends one JSON object as a single line to the log file.
    'a' mode = append, never overwrite. This runs on EVERY request,
    success or failure, so the log file grows over time as a permanent
    record of everything the API has ever been asked.
    """
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    This runs whenever Pydantic rejects a request BEFORE it ever reaches
    predict_risk() — e.g. temperature: 600 violating le=55. Without this
    handler, FastAPI would still return a 422 to the caller, but nothing
    would ever get logged, because predict_risk() never starts running.

    request.body() gives us the raw JSON text the client originally sent,
    even though it failed to parse into a valid WeatherInput object.
    """
    raw_body = await request.body()
    try:
        raw_input = json.loads(raw_body)
    except Exception:
        raw_input = raw_body.decode(errors="replace")

    log_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "validation_error",
        "input": raw_input,
        "error": exc.errors(),   # Pydantic's detailed list of what failed and why
    })

    # We still need to return the SAME 422 response FastAPI would have
    # given anyway — logging an error should never change what the
    # caller sees, only add a record of it on our side.
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.post("/predict")
def predict_risk(weather: WeatherInput):
    # time.time() gives a timestamp in seconds (as a float, fractional
    # part = sub-second precision). We capture it BEFORE and AFTER the
    # actual prediction work, so the difference is the latency.
    start_time = time.time()

    try:
        input_df = pd.DataFrame([{
            "temperature": weather.temperature,
            "humidity":    weather.humidity,
            "wind_speed":  weather.wind_speed,
            "rainfall":    weather.rainfall,
        }])

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        class_names = model.classes_

        confidence_by_class = {
            class_names[i]: round(float(probabilities[i]), 3)
            for i in range(len(class_names))
        }

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Log a SUCCESS event — input, output, latency, all in one record
        log_event({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "input": weather.model_dump(),
            "prediction": prediction,
            "confidence": confidence_by_class,
            "latency_ms": latency_ms,
        })

        return {
            "predicted_risk": prediction,
            "confidence": confidence_by_class
        }

    except Exception as e:
        # If ANYTHING above goes wrong, we land here instead of crashing
        # silently. We log the failure with the error message, then
        # re-raise so FastAPI still returns a proper error response to
        # the client — logging an error should never hide it from the caller.
        latency_ms = round((time.time() - start_time) * 1000, 2)

        log_event({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "input": weather.model_dump(),
            "error": str(e),
            "latency_ms": latency_ms,
        })

        raise


@app.get("/health")
def health_check():
    return {"status": "ok"}