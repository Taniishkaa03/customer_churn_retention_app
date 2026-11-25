# Customer Churn Prediction & Intelligent Retention System

An interactive Streamlit application built with Python to analyze telecom customer churn risk, simulate retention interventions, and prioritize high-value at-risk accounts.

## Features
- **EDA & Model Benchmarking:** Explores contract tiers, billing, and support call patterns while comparing Random Forest and Logistic Regression models.
- **Live Diagnostic & What-If Simulator:** Adjust customer tenure, monthly bill, data usage, support tickets, and contract type to see churn risk and estimated post-intervention impact.
- **Priority Queue & ROI Engine:** Ranks at-risk accounts by churn probability and annual customer value to estimate retention budget and potential revenue saved.
- **Data Structures & Complexity Lab:** Demonstrates dictionary-based lookups and heap-based top-K extraction scaling.

## Tech Stack
- **Frontend & UI:** Streamlit, Plotly
- **Data & Machine Learning:** Pandas, NumPy, scikit-learn
- **Data Structures:** Python `heapq`, dictionaries

## How to Run Locally
1. Clone or download this repository and open a terminal in the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
