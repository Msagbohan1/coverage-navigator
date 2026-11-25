# Model Card — Coverage Navigator Demonstration

## Purpose
Estimate whether a scheduled healthcare service may encounter a **preventable coverage issue** before care is delivered.

## Intended Use
- demonstrate human-in-the-loop AI design,
- explore pre-service coverage-risk triage,
- illustrate governance and evaluation considerations.

## Out-of-Scope Use
The model must not be used to:
- make final coverage determinations,
- approve or deny medical care,
- determine medical necessity,
- replace benefit-document review,
- make clinical decisions.

## Data
Synthetic data only. No patient, claims, or protected health information is included.

## Features
Synthetic features include service type, plan type, network status, prior-authorization requirement/status, documentation completeness, diagnosis/coverage-code alignment, benefit-limit indicator, and a historical denial-rate proxy.

## Key Risks
1. Historical outcomes may encode administrative artifacts.
2. Denial labels may be inconsistent across payers.
3. Model performance may vary across groups.
4. Users may over-rely on model output.
5. High alert volume may create workflow fatigue.

## Safeguards
- human verification for every flagged case,
- explanation shown with the risk flag,
- subgroup performance monitoring,
- abstention/escalation for uncertain cases,
- audit logs,
- regular data-quality review.

## Evaluation
A real pilot should track precision/recall, preventable issue reduction, subgroup performance, alert utilization, staff override rates, resolution time, patient experience, and privacy/governance compliance.

## Limitation
This model is a synthetic demonstration. Its performance cannot be generalized to real claims operations.

## Measured Performance (Synthetic Data)

Trained with `src/train_model.py` on a 75/25 split. Held-out test set: n=450.

**ROC AUC: 0.77**

The single most important design choice is the alert threshold:

| Threshold | Recall | Precision | Share of cases flagged |
|---|---|---|---|
| 0.15 | 0.87 | 0.35 | 0.60 |
| 0.20 | 0.76 | 0.40 | 0.46 |
| 0.25 | 0.66 | 0.42 | 0.38 |
| 0.30 | 0.56 | 0.45 | 0.30 |
| 0.40 | 0.49 | 0.58 | 0.20 |
| 0.50 | 0.39 | 0.66 | 0.14 |
| 0.60 | 0.22 | 0.75 | 0.07 |

The demo uses **0.50** and deliberately favors precision over recall. At 0.50,
two out of three alerts are real issues and 14% of cases go to review. Lower
thresholds catch more issues, but most of their alerts are false alarms, and
staff who see mostly false alarms stop trusting the system. An ignored alert
prevents nothing.

The cost of this choice is stated openly: at 0.50 the model misses roughly
three out of five preventable issues, and the missed cases may not be spread
evenly across patient groups. Subgroup monitoring and the low-confidence
escalation path exist to watch for that.

The threshold is not a hyperparameter to be tuned for a leaderboard metric. It
determines how many specialists an insurer must staff, and it encodes an
institutional judgment about the cost of a missed coverage issue versus an
unnecessary review. That judgment belongs to the workflow owner and should be
documented, revisited during the pilot, and audited, not inherited from a
library default.

## Abstention

The model abstains instead of scoring when the case involves a new insurance
product with limited history, missing or outdated patient history, or
conflicting documentation. Those cases are escalated to a human early.

## Subgroup Results and Why They Are Not a Fairness Claim

| Group | n (test) | Base rate | ROC AUC | Stable estimate |
|---|---|---|---|---|
| English | 331 | 0.23 | 0.74 | yes |
| Spanish | 53 | 0.23 | 0.95 | no |
| Other | 39 | 0.36 | 0.81 | no |
| French | 27 | 0.22 | 0.80 | no |

Measured AUC ranges from 0.74 to 0.95 across language groups. That spread is
not evidence that the model serves some groups better than others: three of the
four groups have fewer than 60 test cases, and estimates from cells that small
move substantially with a different random seed.

The operational lesson is that a fairness monitoring program needs a minimum
subgroup sample size defined **before** results are reviewed. Without one, a
monitoring dashboard will produce a stream of apparent disparities that are
noise, and teams will either chase them or learn to ignore the dashboard
entirely. Both failure modes are worse than not measuring.

The elevated base rate in the "Other" group (0.36 versus roughly 0.23) is
deliberately built into the synthetic data generator so that subgroup
monitoring has a real signal to detect. It is not an empirical finding about
any real population.
