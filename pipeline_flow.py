# pipeline_flow.py
#
# This wraps your existing data-generation and training logic into a
# PREFECT FLOW. Think of it as the PBS-script-equivalent: instead of you
# manually running two separate python files in order, this defines the
# whole pipeline as code, and Prefect tracks every run — when it started,
# whether it succeeded, how long each step took, and lets you retry just
# the failed step instead of the whole thing.
#
# Run it directly (one-time, manual trigger) with:
#   python3 pipeline_flow.py
#
# We'll add automatic scheduling in the next step, AFTER confirming this
# runs correctly on its own first.

from prefect import flow, task
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ── TASK 1: Generate the dataset ──────────────────────────────
# @task marks this function as a trackable unit of work. Prefect logs
# when it starts, when it finishes, and whether it raised an error —
# similar to how a single PBS job's status gets recorded, just for
# one STEP instead of one whole script.
@task(name="generate_data", retries=1, retry_delay_seconds=5)
def generate_data() -> str:
    """
    Same logic as generate_data.py, just living inside a Prefect task now.
    retries=1 means: if this fails (e.g. a transient file-system hiccup),
    Prefect automatically tries ONE more time before giving up — this is
    the orchestration-layer equivalent of a PBS job retry, except PBS
    doesn't actually do this for you automatically; you'd resubmit by hand.
    """
    np.random.seed(42)
    N = 500

    temperature = np.random.uniform(10, 45, N)
    humidity    = np.random.uniform(5, 100, N)
    wind_speed  = np.random.uniform(0, 60, N)
    rainfall    = np.random.uniform(0, 50, N)

    risk_score = (
        (temperature / 45) * 0.4 +
        ((100 - humidity) / 100) * 0.3 +
        (wind_speed / 60) * 0.2 +
        ((50 - rainfall) / 50) * 0.1
    )
    risk_score += np.random.normal(0, 0.05, N)

    risk_category = pd.cut(
        risk_score,
        bins=[-np.inf, 0.4, 0.65, np.inf],
        labels=["Low", "Medium", "High"]
    )

    df = pd.DataFrame({
        "temperature": temperature.round(1),
        "humidity":    humidity.round(1),
        "wind_speed":  wind_speed.round(1),
        "rainfall":    rainfall.round(1),
        "risk_category": risk_category
    })

    output_path = "fire_risk_data.csv"
    df.to_csv(output_path, index=False)

    print(f"[generate_data] Wrote {len(df)} rows to {output_path}")
    return output_path   # we return the path so the next task can use it


# ── TASK 2: Train the model ────────────────────────────────────
@task(name="train_model", retries=1, retry_delay_seconds=5)
def train_model(data_path: str) -> dict:
    """
    Takes the path returned by generate_data() as an argument. This is
    how Prefect tasks chain together — not by guessing file names, but
    by explicitly passing outputs from one task as inputs to the next.
    """
    df = pd.read_csv(data_path)

    X = df[["temperature", "humidity", "wind_speed", "rainfall"]]
    y = df["risk_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    model_path = "fire_risk_model.joblib"
    joblib.dump(model, model_path)

    print(f"[train_model] Accuracy: {accuracy:.2%}")
    print(f"[train_model] Saved model to {model_path}")

    # Return a small summary dict — useful for logging/inspection later,
    # and this is exactly the kind of thing a real monitoring dashboard
    # would track run-over-run to catch accuracy DROPPING over time.
    return {
        "accuracy": accuracy,
        "model_path": model_path,
        "high_recall": report["High"]["recall"],
    }


# ── THE FLOW: ties the tasks together in order ────────────────
# @flow marks the top-level function. Calling generate_data() and
# train_model() INSIDE a flow function is what makes Prefect track
# them as a connected pipeline, not two unrelated function calls.
@flow(name="fire-risk-training-pipeline")
def training_pipeline():
    data_path = generate_data()
    result = train_model(data_path)

    print(f"\nPipeline complete.")
    print(f"  Accuracy:    {result['accuracy']:.2%}")
    print(f"  High recall: {result['high_recall']:.2%}")

    return result


# ── Entry point — runs the flow once when this file is executed ──
if __name__ == "__main__":
    training_pipeline()