"""
Train a demonstration model on synthetic Coverage Navigator data.

This script is transparent and lightweight. It is not intended for production
use or real coverage decisions.

Beyond a single accuracy figure, it records the operating characteristics that
actually matter for a review queue: how recall, precision, and reviewer
workload trade off against the alert threshold, and how the model behaves
across language subgroups.

Usage:
    python src/train_model.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "synthetic_coverage_cases.csv")

target = "preventable_coverage_issue"
features = [
    "service_type", "plan_type", "network_status",
    "prior_auth_required", "prior_auth_on_file",
    "documentation_complete", "diagnosis_code_match",
    "benefit_limit_flag", "historical_denial_rate",
]

X = df[features]
y = df[target]

categorical = ["service_type", "plan_type", "network_status"]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
    "at_default_threshold_0_5": {
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 3),
    },
}

# --- Threshold sweep -------------------------------------------------------
# The threshold is not a technical detail. It decides how many appointments a
# claims specialist must review, so it is a staffing and workflow decision as
# much as a modelling one.
sweep = []
for t in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
    p = (proba >= t).astype(int)
    sweep.append({
        "threshold": t,
        "recall": round(float(recall_score(y_test, p, zero_division=0)), 3),
        "precision": round(float(precision_score(y_test, p, zero_division=0)), 3),
        "share_of_cases_flagged": round(float(p.mean()), 3),
    })
metrics["threshold_sweep"] = sweep

# --- Subgroup performance --------------------------------------------------
# Reported with group sizes, because small cells produce unstable estimates
# that should not be read as evidence of fairness or unfairness.
test = df.loc[X_test.index].copy()
test["_proba"] = proba
subgroups = {}
for group, sub in test.groupby("language_group"):
    entry = {
        "n": int(len(sub)),
        "base_rate": round(float(sub[target].mean()), 3),
    }
    if sub[target].nunique() > 1:
        entry["roc_auc"] = round(float(roc_auc_score(sub[target], sub["_proba"])), 3)
    else:
        entry["roc_auc"] = None
    entry["estimate_is_stable"] = bool(entry["n"] >= 100)
    subgroups[group] = entry
metrics["subgroups"] = subgroups

joblib.dump(model, MODEL_DIR / "coverage_navigator_demo.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

print(json.dumps(metrics, indent=2))
print("\nWrote model/coverage_navigator_demo.joblib and model/metrics.json")
print(
    "\nNote: at the default 0.5 threshold this model misses most positive cases.\n"
    "That is the point of the sweep above - see model/model_card.md."
)
