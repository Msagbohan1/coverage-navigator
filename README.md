# Coverage Navigator

## AI-Assisted Pre-Service Coverage Decision Support

**Coverage Navigator** is a portfolio project developed from a semester project in Brown University's Data Science, Policy, Governance, and Society program.

The project asks:

> **Can AI help identify preventable health-insurance coverage problems before care is delivered—without replacing human judgment?**

Coverage Navigator is designed as a **human-in-the-loop decision-support concept**. The model forecasts whether a scheduled service is likely to encounter a preventable coverage issue, explains the main reasons for the flag, and routes the case to an insurer claims specialist. The specialist verifies the recommendation against the member's benefit documents and makes the final determination.

**Live demo:** https://coverage-navigator-f47ewhli4ni6mrqmqhlzxz.streamlit.app/
**Project page:** https://msagbohan1.github.io/coverage-navigator/

This repository expands the original academic work into a public-facing portfolio project with a synthetic dataset, exploratory visualizations, a machine-learning demonstration, a Streamlit prototype, governance documentation, pilot design, and GitHub Pages project page.

> **Important:** This is an academic demonstration using synthetic data. It is not a clinical tool, insurance product, or production coverage-determination system.

**Who it is for:** insurer claims teams, hospital intake and prior-authorization staff, analysts, and the CMS and insurer leaders who would approve a pilot. It is a staff-side tool. A patient-facing chatbot was considered and rejected, because it would shift the work of navigating coverage onto patients with limited health literacy, limited English proficiency, or limited digital access.

---

## The Policy Problem

| | |
|---|---|
| **100 million** | Americans carry medical debt (Levey, 2022) |
| **14%** | of privately insured adults correctly understand their health insurance (Loewenstein et al., 2013) |
| **<1%** | of denied claims are appealed (Claxton et al., 2025) |

The current system finds coverage problems **after** care: the claim is denied, the patient receives an unexpected bill, and almost no one appeals. Coverage Navigator finds them **before** care.

The project began with medical debt and was narrowed to the **health-insurance understanding gap**: the difference between what insured patients believe their plan covers and what the plan actually pays.

Post-denial appeal support was considered and rejected. It is reactive, the financial harm has already begun by the time a claim is denied, and no insurer would credibly fund a system built to overturn its own decisions.

Coverage Navigator moves the intervention upstream by identifying potential issues before care, including:
- missing prior authorization,
- provider-network problems,
- benefit limitations,
- incomplete documentation,
- coding / coverage mismatches,
- patterns associated with elevated historical denial rates.

## Proposed Workflow

```text
Appointment scheduled (hospital or clinic)
        ↓
Coverage Navigator reviews the case in the background
        ↓
   ┌────────────────┬─────────────────────────┬──────────────────────────────┐
 Low risk        High risk               Low confidence
 → proceeds      → insurer specialist    (new product, missing history,
   as normal       queue                  conflicting documentation)
                                          → escalated, model abstains
        ↓
Insurer claims specialist verifies against benefit documents
and agrees, or overrides with a reason
        ↓
Specialist contacts the hospital's intake / prior-authorization team
        ↓
Hospital submits what is missing; insurer prior-auth team reviews it
        ↓
Coverage clarified → patient receives care
```

Who does what:
- **The hospital or clinic** schedules care, and its intake and prior-authorization staff submit requests and fix documentation.
- **The insurer** employs the claims specialist who reviews flags, and runs its own prior-authorization review.
- **The model** finds similar past cases and patterns, flags risk, and explains why. It does not read plan rules and does not make coverage decisions.

In a real pilot, Coverage Navigator would sit inside the intake and prior-authorization systems staff already use, not in a separate portal. A tool outside daily workflow is unlikely to be adopted, whatever its accuracy. The Streamlit app is a standalone demo of what that integrated view would show.

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

A limitation the synthetic data hides: real denial labels are noisy. In public marketplace claims data, the most common recorded denial reason is a vague "other," at roughly a third of denials. Claims records are also collected for payment, not research, and a medication denied under one diagnosis code may be covered under a more appropriate one. A model trained on these outcomes can learn billing habits rather than true coverage risk.

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
| Unequal model performance | Some groups may have less historical data | Subgroup evaluation, funded multilingual outreach, fairness monitoring |
| Thin or unreliable data | New plan products, missing history, or conflicting documentation make a score untrustworthy | The model abstains and the case is escalated to a human early |
| Automation bias | Staff may over-trust an AI recommendation | Human verification, explanations, training, audit logs |
| Data sharing & privacy | Cross-payer/provider data may be sensitive | Minimum-necessary data, access controls, encryption, governance |
| Workflow fatigue | Too many false alerts may be ignored | Favor useful, high-confidence alerts; monitor adoption |

## What the Model Shows

The demonstration model is a logistic regression trained on the synthetic dataset (`src/train_model.py`). On a held-out test set it reaches an **ROC AUC of 0.77** — and, at the default classification threshold of 0.50, it misses about three out of five preventable issues.

| Threshold | Recall | Precision | Share of cases sent to review |
|---|---|---|---|
| 0.20 | 0.76 | 0.40 | 46% |
| 0.30 | 0.56 | 0.45 | 30% |
| 0.40 | 0.49 | 0.58 | 20% |
| **0.50 (default)** | **0.39** | **0.66** | **14%** |

**The design deliberately favors precision over recall.** Trust, not information, is the scarce resource. From experience working in health insurance claims: frontline staff quickly stop paying attention to a system that raises too many false alarms, and once they do, it effectively stops existing.

At 0.50, two out of three alerts are real issues and 14% of cases go to review. Catching three-quarters of issues would mean reviewing nearly half of all appointments, with more than half of those alerts false. That is the alert fatigue the design is meant to avoid.

The trade-off is stated openly: at 0.50, roughly three in five preventable issues are not flagged, and those missed cases may not be spread evenly across patient groups. The design answers this with subgroup monitoring and an escalation path for low-confidence cases, not with more alerts.

The threshold is still an institutional judgment, not a modelling one. It belongs to the workflow owner, should be documented and revisited during the pilot, and should never be inherited from a library default. The demo exposes it as a control so the trade-off can be seen.

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

The recommendation is an **18-month pilot across 25 hospitals and 500 clinics**, with continuous fairness audits, human oversight at every step, and expansion only if the pilot succeeds.

1. **Months 1–6 — Planning & design**
   - finalize governance and data-sharing agreements
   - configure the system and baseline measurement
   - partner onboarding and staff training

2. **Months 7–12 — Integration & go-live**
   - integrate with intake and prior-authorization workflows
   - go live at pilot sites
   - monitor performance, alert use, and user feedback

3. **Months 13–18 — Evaluation & decision**
   - measure outcomes and equity
   - refine the model and workflows
   - report results to CMS and insurers
   - decide to scale, adapt, or stop

Because the 25-hospital scope is ambitious, each phase ends with a governance check before the next one starts. Integration with multiple EHR and practice-management systems is likely to be the slowest part.

## Who Decides: CMS, Insurers, and Providers

The pilot has two executive audiences, **CMS and participating insurers**, and their incentives conflict. Preventing denials improves patient outcomes, but it can also raise insurer spending by stopping denials that would otherwise have saved them money.

Insurers are likely to participate only if fewer appeals, less rework, better member satisfaction, regulatory goodwill, and long-term retention outweigh the additional claims costs. If that balance shifts during the pilot, they may slow data sharing, delay integration, or withdraw, even if the model performs well. Making a credible business case for insurer participation is therefore as important as model performance.

Hospitals and clinics are the third party. Their intake and prior-authorization teams have to act on alerts before care, so the tool has to fit their workflow, not add a new one.

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

A real pilot would require bottom-up costing from participating organizations. The threshold analysis above makes claims-specialist time the dominant variable cost: moving from the 0.50 default to a high-recall threshold such as 0.20 would roughly triple review volume.

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
- **Pilot scale** — the 25-hospital, 500-clinic pilot keeps its three phases but adds a governance check between each one.

This version also adds what the coursework did not have: a working model, the threshold analysis above, a low-confidence escalation path, a specialist queue that records agreements and overrides, and a demo where the governance trade-off can be manipulated rather than only described.

## References

- Claxton et al. (2025). Cited for: fewer than 1% of denied claims are appealed. See also KFF, *Claims Denials and Appeals in ACA Marketplace Plans in 2023* (2025): https://www.kff.org/private-insurance/claims-denials-and-appeals-in-aca-marketplace-plans-in-2023/
- Desai, P., et al. (2024). *Findings.* Agency for Healthcare Research and Quality (US). https://www.ncbi.nlm.nih.gov/books/NBK617038/
- Levey, N. N. (2022). Cited for: 100 million Americans carry medical debt. KFF Health News.
- Loewenstein, G., et al. (2013). Consumers' misunderstanding of health insurance. *Journal of Health Economics*, 32(5), 850–862.

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
