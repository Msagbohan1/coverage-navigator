import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Coverage Navigator", layout="wide")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synthetic_coverage_cases.csv"
MODEL_PATH = ROOT / "model" / "coverage_navigator_demo.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    """Return the trained pipeline, or None if it has not been trained yet."""
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


@st.cache_data
def load_metrics(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


if not DATA_PATH.exists():
    st.error(
        f"Cannot find the synthetic dataset at `{DATA_PATH.relative_to(ROOT)}`. "
        "Clone the full repository, or regenerate the file, then reload this page."
    )
    st.stop()

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

FEATURES = [
    "service_type", "plan_type", "network_status",
    "prior_auth_required", "prior_auth_on_file",
    "documentation_complete", "diagnosis_code_match",
    "benefit_limit_flag", "historical_denial_rate",
]


def rule_based_risk(row):
    """Transparent fallback scorecard used when no trained model is present.

    The weights are hand-set, not learned. They are legible on purpose so the
    demo still explains itself before anyone runs the training script.
    """
    logit = (
        -2.8
        + 1.8 * int(row["prior_auth_required"] and not row["prior_auth_on_file"])
        + 1.5 * int(row["network_status"] == "Out of Network")
        + 1.1 * int(not row["documentation_complete"])
        + 1.3 * int(not row["diagnosis_code_match"])
        + 1.1 * int(row["benefit_limit_flag"])
        + 2.0 * row["historical_denial_rate"]
        + 0.35 * int(row["service_type"] == "Specialty Medication")
        + 0.20 * int(row["plan_type"] == "HMO")
    )
    return 1 / (1 + np.exp(-logit))


st.title("Coverage Navigator")
st.caption("AI-assisted pre-service coverage decision support — synthetic portfolio demo")

st.warning(
    "Academic demonstration only. This prototype uses synthetic data and must not be used "
    "for real insurance, medical-necessity, or clinical decisions."
)

if model is None:
    st.sidebar.info(
        "Showing the rule-based scorecard. Run `python src/train_model.py` to load the "
        "trained model instead."
    )
else:
    st.sidebar.success("Trained model loaded.")

page = st.sidebar.radio(
    "Explore",
    ["Overview", "Coverage Review", "Specialist Dashboard", "Model & Governance"],
)


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
if page == "Overview":
    st.header("Project Overview")
    st.write(
        "Coverage Navigator forecasts whether a scheduled service may encounter a preventable "
        "coverage issue before care is delivered. The system flags risk and explains the likely "
        "drivers; a claims specialist verifies the case against benefit documents and makes the "
        "final call."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Synthetic cases", f"{len(df):,}")
    c2.metric("Synthetic issue rate", f"{df['preventable_coverage_issue'].mean():.1%}")
    c3.metric("Human final decision", "Required")

    st.subheader("Conceptual workflow")
    st.code(
        "Appointment → Coverage review → AI risk forecast → Explanation → "
        "Claims specialist → Verify benefits → Resolve / escalate"
    )

    st.subheader("What success means")
    st.write(
        "The project evaluates more than model accuracy: fairness across patient groups, "
        "staff adoption, workflow integration, patient experience, and governance are part "
        "of the intervention."
    )


# --------------------------------------------------------------------------
# Coverage Review
# --------------------------------------------------------------------------
elif page == "Coverage Review":
    st.header("Synthetic Coverage Review")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Case details")
        service = st.selectbox("Service type", sorted(df["service_type"].unique()))
        plan = st.selectbox("Plan type", sorted(df["plan_type"].unique()))
        network = st.selectbox("Network status", ["In Network", "Out of Network"])
        pa_required = st.checkbox("Prior authorization required")
        pa_on_file = st.checkbox("Prior authorization on file", value=True)
        documentation = st.checkbox("Documentation complete", value=True)
        code_match = st.checkbox("Diagnosis / coverage-code match", value=True)
        benefit_limit = st.checkbox("Potential benefit limit")
        hist = st.slider("Historical denial-rate proxy", 0.01, 0.60, 0.18, 0.01)

        threshold = st.slider(
            "Alert threshold",
            0.10, 0.90, 0.30, 0.05,
            help="Cases at or above this risk score are routed to a specialist. "
                 "Lowering it catches more issues and creates more review work.",
        )

    case = pd.DataFrame([{
        "service_type": service,
        "plan_type": plan,
        "network_status": network,
        "prior_auth_required": int(pa_required),
        "prior_auth_on_file": int(pa_on_file),
        "documentation_complete": int(documentation),
        "diagnosis_code_match": int(code_match),
        "benefit_limit_flag": int(benefit_limit),
        "historical_denial_rate": hist,
    }])

    if model is not None:
        risk = float(model.predict_proba(case[FEATURES])[0, 1])
        source = "Trained logistic-regression model"
    else:
        risk = float(rule_based_risk(case.iloc[0]))
        source = "Rule-based scorecard (model not trained yet)"

    with right:
        st.subheader("AI-assisted forecast")
        st.metric("Synthetic coverage-issue risk", f"{risk:.0%}")
        st.caption(f"Source: {source}")

        reasons = []
        if pa_required and not pa_on_file:
            reasons.append("Prior authorization may be missing.")
        if network == "Out of Network":
            reasons.append("Provider appears out of network.")
        if not documentation:
            reasons.append("Documentation is incomplete.")
        if not code_match:
            reasons.append("Diagnosis / coverage-code mismatch.")
        if benefit_limit:
            reasons.append("A benefit limitation may apply.")
        if hist > 0.30:
            reasons.append("Similar historical cases show elevated denial rates.")

        if risk >= threshold:
            st.error(f"Routed for review — risk is at or above the {threshold:.0%} threshold.")
        else:
            st.success(f"Below the {threshold:.0%} threshold — not routed in this scenario.")

        if reasons:
            st.write("**Contributing factors**")
            for r in reasons:
                st.write("• " + r)
        else:
            st.write("No rule-based contributing factors identified.")

        if risk >= threshold:
            st.info(
                "Recommended action: route to a claims specialist for verification against the "
                "member's benefit documents before care."
            )

    st.caption(
        "Forecast only — not automated decision-making. A human specialist remains responsible "
        "for verification and the final determination. Note that a case can score below the "
        "threshold and still carry a real issue; the threshold controls workload, not truth."
    )


# --------------------------------------------------------------------------
# Specialist Dashboard
# --------------------------------------------------------------------------
elif page == "Specialist Dashboard":
    st.header("Specialist Dashboard")
    st.write("Synthetic queue of cases prioritized for human review.")

    threshold = st.slider("Show cases above risk score", 0.10, 0.90, 0.30, 0.05)

    scored = df.copy()
    if model is not None:
        scored["risk"] = model.predict_proba(scored[FEATURES])[:, 1]
        score_label = "Model risk score"
    else:
        scored["risk"] = scored["synthetic_risk_score"]
        score_label = "Stored synthetic risk score"

    queue = scored[scored["risk"] >= threshold].sort_values("risk", ascending=False)

    caught = queue["preventable_coverage_issue"].sum()
    total_issues = scored["preventable_coverage_issue"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Cases in queue", f"{len(queue):,}", f"{len(queue) / len(scored):.0%} of all cases")
    c2.metric("Issues captured", f"{caught:,}", f"{caught / total_issues:.0%} of known issues")
    c3.metric(
        "Queue precision",
        f"{caught / len(queue):.0%}" if len(queue) else "—",
        help="Share of queued cases that are true issues in the synthetic data.",
    )

    st.caption(
        f"Scores from: {score_label}. Lowering the threshold captures more issues but enlarges "
        "the queue — the trade-off a real deployment has to staff for."
    )

    display = queue.head(50).copy()
    display["risk"] = display["risk"].round(3)
    st.dataframe(
        display[[
            "case_id", "service_type", "plan_type", "network_status",
            "risk", "explanation",
        ]],
        width="stretch",
    )

    st.caption(
        "In a real implementation, reviewers would verify benefit documents, document an action, "
        "and create an auditable record of overrides and escalations."
    )


# --------------------------------------------------------------------------
# Model & Governance
# --------------------------------------------------------------------------
elif page == "Model & Governance":
    st.header("Model & Governance")

    left, right = st.columns(2)

    with left:
        st.subheader("Model inputs")
        st.write(
            """
            - Plan / benefit information
            - Prior-authorization requirements
            - Network status
            - Documentation completeness
            - Diagnosis / coverage-code alignment
            - Historical claim patterns
            - Service type
            """
        )

        st.subheader("Model role")
        st.write(
            "The model forecasts risk and provides an explanation. It does not make a final "
            "coverage determination."
        )

    with right:
        st.subheader("Governance controls")
        st.write(
            """
            - Human review for every flagged case
            - Minimum-necessary data access
            - Role-based access controls
            - Audit logs
            - Data-quality monitoring
            - Subgroup performance review
            - Abstention / escalation for uncertain cases
            """
        )

        st.subheader("Pilot evaluation")
        st.write(
            """
            - Preventable issue reduction
            - Precision / recall
            - Fairness across groups
            - Staff adoption
            - Workflow integration
            - Patient experience
            - Privacy compliance
            """
        )

    if metrics:
        st.subheader("Where the threshold sets the workload")
        st.write(
            "Choosing an alert threshold decides how much of the caseload a specialist team "
            "must review. It is a staffing decision as much as a modelling one."
        )
        sweep = pd.DataFrame(metrics["threshold_sweep"])
        sweep.columns = ["Threshold", "Recall", "Precision", "Share of cases flagged"]
        st.dataframe(sweep, width="stretch", hide_index=True)
        st.caption(
            f"Held-out test set, n={metrics['n_test']}. ROC AUC {metrics['roc_auc']}. "
            "At the default 0.5 threshold this model misses most true issues — which is why "
            "the threshold belongs to the workflow owner, not the default setting."
        )

        st.subheader("Subgroup performance")
        subs = pd.DataFrame(metrics["subgroups"]).T.reset_index()
        subs.columns = ["Group", "n", "Base rate", "ROC AUC", "Stable estimate"]
        st.dataframe(subs, width="stretch", hide_index=True)
        st.caption(
            "Group sizes differ a great deal. Differences measured on small cells are unstable "
            "and should not be read as evidence of fairness or unfairness — establishing "
            "minimum subgroup sample sizes is itself a governance requirement."
        )
    else:
        st.info("Run `python src/train_model.py` to populate model metrics on this page.")

    st.subheader("Synthetic monitoring example")
    subgroup = df.groupby("language_group")["preventable_coverage_issue"].mean()
    st.bar_chart(subgroup)
    st.caption(
        "The elevated rate in the 'Other' group is deliberately built into the synthetic data "
        "generator so that subgroup monitoring has something to detect."
    )
