"""ML Zoomcamp 2026, Homework 1: reproducible answers.

Run with Python after installing pandas and numpy:
    python -m pip install pandas numpy
    python hw01.py

Data: DataTalksClub/machine-learning-zoomcamp, pinned 2026 car release.
"""

import numpy as np
import pandas as pd


DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)

df = pd.read_csv(DATA_URL)
asia = df.loc[df["origin"] == "Asia"]

# Q1: report the Pandas version in the environment running this file.
print(f"Q1 Pandas version: {pd.__version__}")

# Q2: one row corresponds to one record.
print(f"Q2 Records count: {len(df)}")

# Q3: number of distinct non-missing fuel types.
print(f"Q3 Fuel types: {df['fuel_type'].nunique()}")

# Q4: count columns containing at least one missing value.
print(f"Q4 Columns with missing values: {df.isna().any().sum()}")

# Q5: the highest MPG among cars from Asia.
print(f"Q5 Max fuel efficiency in Asia: {asia['fuel_efficiency_mpg'].max()}")

# Q6: compare the original median with the median after filling NaN with mode.
hp = df["horsepower"]
before = hp.median()
most_frequent = hp.mode().iloc[0]
after = hp.fillna(most_frequent).median()
change = "Yes, it increased" if after > before else (
    "Yes, it decreased" if after < before else "No"
)
print(f"Q6 Median: {before} -> {after}; answer: {change}")

# Q7: preserve the dataset order and the specified order of the two columns.
X = asia[["vehicle_weight", "model_year"]].head(7).to_numpy()
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
XTX = X.T @ X
w = np.linalg.inv(XTX) @ X.T @ y
print(f"Q7 Sum of weights: {w.sum():.9f}; closest option: 0.369")
