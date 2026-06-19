# generate_data.py
#
# Creates a small synthetic dataset: weather conditions -> fire risk category.
# We're making up the data with simple rules + randomness, because the point
# of this project is everything AROUND the model, not the data itself.

import pandas as pd
import numpy as np

# Set a seed so you get the SAME random data every time you run this.
# Without this, your numbers would be different each run, which makes
# debugging confusing later (the model's accuracy would shift each time).
np.random.seed(42)

# How many rows of fake data to generate
N = 500

# ── Generate the 4 input features ──────────────────────────
# np.random.uniform(low, high, N) gives N random numbers between low and high

temperature = np.random.uniform(10, 45, N)     # Celsius
humidity    = np.random.uniform(5, 100, N)     # percent
wind_speed  = np.random.uniform(0, 60, N)       # km/h
rainfall    = np.random.uniform(0, 50, N)       # mm in last 24h

# ── Define a simple "risk score" rule ──────────────────────
# This is NOT real fire science — it's a made-up formula so that
# high temp + low humidity + high wind + low rain = higher risk.
# We add random noise so it's not a perfectly clean pattern (more realistic).

risk_score = (
    (temperature / 45) * 0.4 +
    ((100 - humidity) / 100) * 0.3 +
    (wind_speed / 60) * 0.2 +
    ((50 - rainfall) / 50) * 0.1
)
risk_score += np.random.normal(0, 0.05, N)   # small random noise

# ── Convert the continuous score into 3 categories ─────────
# pd.cut slices a numeric range into labeled buckets.
risk_category = pd.cut(
    risk_score,
    bins=[-np.inf, 0.4, 0.65, np.inf],
    labels=["Low", "Medium", "High"]
)

# ── Assemble into a DataFrame (a table, like an Excel sheet) ─
df = pd.DataFrame({
    "temperature": temperature.round(1),
    "humidity":    humidity.round(1),
    "wind_speed":  wind_speed.round(1),
    "rainfall":    rainfall.round(1),
    "risk_category": risk_category
})

# ── Save to a CSV file so other scripts can load it ─────────
df.to_csv("fire_risk_data.csv", index=False)

print(f"Generated {len(df)} rows.")
print(df["risk_category"].value_counts())
print("\nFirst 5 rows:")
print(df.head())