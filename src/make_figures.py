"""
Regenerate the figures used on the project page.

Every image in docs/images/ is produced by this script, so the visuals in the
repository can be reproduced from the synthetic dataset rather than trusted.

Usage:
    python src/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_coverage_cases.csv"
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# 1. Issue rate by service type
rates = df.groupby("service_type")["preventable_coverage_issue"].mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(rates.index, rates.values)
ax.set_xlabel("Issue rate")
ax.set_title("Synthetic Preventable Coverage-Issue Rate by Service Type")
save(fig, "issue_rate_by_service.png")


# 2. Distribution of risk scores
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["synthetic_risk_score"], bins=25)
ax.set_xlabel("Synthetic risk score")
ax.set_ylabel("Number of cases")
ax.set_title("Distribution of Synthetic Coverage-Risk Scores")
save(fig, "risk_score_distribution.png")


# 3. Subgroup monitoring
sub = df.groupby("language_group")["preventable_coverage_issue"].mean().sort_values()
counts = df["language_group"].value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(sub.index, sub.values)
for i, (group, rate) in enumerate(sub.items()):
    ax.text(i, rate + 0.005, f"n={counts[group]}", ha="center", fontsize=9)
ax.set_xlabel("Language group")
ax.set_ylabel("Issue rate")
ax.set_title("Synthetic Outcome Rate by Language Group")
save(fig, "subgroup_monitoring.png")

print("\nGroup sizes differ substantially. Subgroup rates from small cells are")
print("noisy, which is why the model card treats them as a monitoring signal")
print("rather than a fairness conclusion.")
