# train_model.py
#
# Loads the CSV we generated, trains a RandomForestClassifier on it,
# and SAVES the trained model to a file using joblib.
#
# Why save to a file? Right now your model only exists as a variable
# inside this running script. The moment the script ends, it's gone.
# Saving it to disk means ANY other program (an API, a test script,
# anything) can load that exact trained model later without retraining.
# This is the bridge between "I trained something" and "deployment."

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib   # the library used to save/load sklearn models

# ── Load the data we generated in stage 1a ──────────────────
df = pd.read_csv("fire_risk_data.csv")

# Features (inputs) and target (what we're predicting)
X = df[["temperature", "humidity", "wind_speed", "rainfall"]]
y = df["risk_category"]

# ── Split into training data and test data ──────────────────
# We hold back 20% of rows that the model NEVER sees during training,
# so we can honestly check how well it performs on data it hasn't memorized.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train the model ──────────────────────────────────────────
# You already know RandomForest — same idea, tiny dataset.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate on the held-out test set ────────────────────────
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Test accuracy: {accuracy:.2%}")
print("\nDetailed report:")
print(classification_report(y_test, predictions))

# ── Save the trained model to a file ─────────────────────────
# This creates a file called "fire_risk_model.joblib" in this folder.
# It's a serialized (saved) version of the trained model object —
# all its learned tree splits, everything — frozen into a file.
joblib.dump(model, "fire_risk_model.joblib")

print("\nModel saved to fire_risk_model.joblib")