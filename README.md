# Coverage Navigator

## AI-Assisted Pre-Service Coverage Decision Support

**Coverage Navigator** is a portfolio project developed from a semester project in Brown University's Data Science, Policy, Governance, and Society program.

The project asks:

> **Can AI help identify preventable health-insurance coverage problems before care is delivered—without replacing human judgment?**

Coverage Navigator is designed as a **human-in-the-loop decision-support concept**. The model forecasts whether a scheduled service is likely to encounter a preventable coverage issue, explains the main reasons for the flag, and routes the case to a claims specialist. The specialist verifies the recommendation against benefit documents and makes the final determination.

This repository expands the original academic work into a public-facing portfolio project with a synthetic dataset, exploratory visualizations, a machine-learning demonstration, a Streamlit prototype, governance documentation, pilot design, and GitHub Pages project page.

> **Important:** This is an academic demonstration using synthetic data. It is not a clinical tool, insurance product, or production coverage-determination system.

---

## The Policy Problem

The project began with medical debt and was narrowed to the **health-insurance understanding gap**: the difference between what insured patients believe their plan covers and what the plan actually pays.

Coverage Navigator moves the intervention upstream by identifying potential issues before care, including:
- missing prior authorization,
- provider-network problems,
- benefit limitations,
- incomplete documentation,
- coding / coverage mismatches,
- patterns associated with elevated historical denial rates.

## Proposed Workflow

```text
Appointment scheduled
        ↓
Coverage information reviewed
        ↓
AI-assisted risk forecast
        ↓
Potential issue flagged + explanation
        ↓
Claims specialist review
        ↓
Verify against benefit documents
        ↓
Resolve / contact provider / escalate
        ↓
Coverage clarified before care
```

The system supports decisions; it does not make autonomous coverage determinations.

## Model Inputs

The conceptual model uses information available in a pre-service workflow, such as:
- plan and benefit information,
- prior-authorization requirements,
- network status,
- documentation completeness,
- diagnosis / coverage-code alignment,
- historical claim patterns,
- service type.

The public demo uses **synthetic versions** of these variables.

## Why Human-in-the-Loop?

The original project deliberately chose **decision support over automation**. Coverage decisions can affect whether and how patients receive care, while historical claims data may contain administrative artifacts, inconsistent labels, and payer-specific practices.

In Coverage Navigator:
- the model **forecasts** risk,
- the system **explains** the flag,
- a human specialist **verifies** the case,
- the human remains **accountable for the final decision**.

## Responsible AI Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Inconsistent denial labels | Denial reasons are not standardized across payers | Data-quality audits, label harmonization, abstention |
| Unequal model performance | Some groups may have less historical data | Subgroup evaluation, multilingual design, fairness monitoring |
| Automation bias | Staff may over-trust an AI recommendation | Human verification, explanations, training, audit logs |
| Data sharing & privacy | Cross-payer/provider data may be sensitive | Minimum-necessary data, access controls, encryption, governance |
| Workflow fatigue | Too many false alerts may be ignored | Favor useful, high-confidence alerts; monitor adoption |

## What the Model Shows

The demonstration model is a logistic regression trained on the synthetic dataset (`src/train_model.py`). On a held-out test set it reaches an **ROC AUC of 0.77** — and, at the default classification threshold of 0.50, it misses about three out of five preventable issues.

That is not a result to hide. It is the clearest evidence for this project's central argument.

| Threshold | Recall | Precision | Share of cases sent to review |
|---|---|---|---|
| 0.20 | 0.76 | 0.40 | 46% |
| 0.30 | 0.56 | 0.45 | 30% |
| 0.40 | 0.49 | 0.58 | 20% |
| 0.50 | 0.39 | 0.66 | 14% |

A prevention system needs high recall, and high recall here means routing between a third and a half of all appointments to a claims specialist. The threshold therefore decides staffing levels and encodes an institutional judgment about the relative cost of a missed coverage issue versus an unnecessary review.

**That judgment is not a modelling decision.** It belongs to the workflow owner, it should be documented and periodically revisited, and it should never be inherited from a library default. The Streamlit demo exposes the threshold as a control for exactly this reason.

Subgroup results carry a parallel lesson. Measured AUC ranges from 0.74 to 0.95 across language groups, but three of the four groups have fewer than 60 test cases — small enough that the spread is mostly noise. A fairness monitoring program needs a minimum subgroup sample size defined before results are reviewed, or it will generate apparent disparities that teams either chase needlessly or learn to ignore. Full figures are in `model/model_card.md`.

## Evaluation: More Than Accuracy

A successful pilot should measure:
- reduction in preventable coverage issues,
- precision and recall,
- fairness across patient groups,
- staff adoption,
- workflow integration,
- patient experience,
- escalation and resolution time,
- privacy and governance compliance.

## Pilot Design

The original proposal asked leadership to approve an **18-month pilot across 25 hospitals and 500 clinics**. That scope was ambitious, so the portfolio version reframes it as a phased rollout.

1. **Months 0–3 — Setup**
   - data agreements
   - privacy review
   - workflow mapping
   - baseline measurement

2. **Months 4–8 — Limited launch**
   - selected hospitals and clinics
   - staff training
   - model calibration
   - alert-utility monitoring

3. **Months 9–14 — Expansion**
   - broader site coverage
   - subgroup performance review
   - integration improvements

4. **Months 15–18 — Evaluation**
   - outcome analysis
   - cost/benefit review
   - governance assessment
   - scale/adapt/stop decision

## Why Now?

- healthcare organizations are rapidly adopting AI-assisted tools,
- governance expectations are developing at the same time as deployment,
- prior-authorization and coverage workflows remain administratively complex,
- responsible workflow design is easier to establish before opaque automation becomes normalized.

The question is not simply whether AI will enter administrative healthcare workflows, but **under what governance conditions it should be allowed to assist decisions**.

## Cost & Resource Considerations

This repository does **not invent a dollar estimate**. Instead, it identifies the main cost drivers:
- data engineering,
- EHR / practice-management integration,
- model development and monitoring,
- cybersecurity and privacy review,
- claims-specialist time,
- staff training,
- program management,
- fairness and performance audits,
- governance and legal review.

A real pilot would require bottom-up costing from participating organizations. Note that the threshold analysis above makes claims-specialist time the dominant variable cost: a threshold chosen for high recall multiplies review volume several times over.

## Repository Structure

```text
coverage-navigator/
├── README.md
├── QUICKSTART.md
├── GITHUB_SETUP.md
├── LICENSE
├── .gitignore
├── app.py
├── requirements.txt
├── data/
│   └── synthetic_coverage_cases.csv
├── model/
│   ├── model_card.md
│   └── metrics.json          # generated by src/train_model.py
├── src/
│   ├── train_model.py
│   └── make_figures.py
├── docs/
│   ├── index.html
│   ├── governance.md
│   ├── pilot.md
│   └── images/
└── notebooks/
    └── README.md
```

## Run the Streamlit Demo

```bash
pip install -r requirements.txt
python src/train_model.py      # trains the model and writes metrics
streamlit run app.py
```

The app runs without the training step, falling back to a transparent rule-based scorecard and saying so in the sidebar. Running training first loads the real model and populates the metrics on the Model & Governance page.

To regenerate the figures used on the project page:

```bash
python src/make_figures.py
```

Publishing instructions are in `GITHUB_SETUP.md`.

## How This Version Differs From the Original Submission

The original coursework was assessed as technically sound in its treatment of forecasting versus automated decision-making, its explicit human verification step, its specific model inputs and risk mitigations, and its attention to competing institutional incentives. Review also identified four gaps, each addressed here:

- **Cost** — the repository names the real cost drivers rather than inventing a dollar figure it cannot support.
- **Why now** — an explicit argument that governance design is easier to establish before opaque automation is normalized.
- **Data sharing and privacy** — treated directly in `docs/governance.md`.
- **Pilot scale** — the original 25-hospital, 500-clinic ask is replaced by a phased rollout with governance gates between stages.

This version also adds what the coursework did not have: a working model, the threshold analysis above, and a demo where the governance trade-off can be manipulated rather than only described.

## Core Takeaway

The machine-learning model is **not the intervention by itself**.

Long-term public value depends on governance, institutional incentives, workflow integration, user trust, accountability, data quality, and the capacity to monitor and adapt the system over time.

Technical performance may justify a pilot. Responsible implementation determines whether it should scale.

## License

MIT — see `LICENSE`. The synthetic dataset contains no real patient, claims, or protected health information.

## Author

**Myriam Sagbohan**
Brown University
Data Science, Policy, Governance, and Society
Summer 2026
