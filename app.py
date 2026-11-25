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
    st.info(
        "**Who this demo is for:** insurer claims teams, hospital intake and prior-authorization "
        "staff, analysts, and the CMS and insurer leaders who would approve a pilot. "
        "It is a staff-side concept, not a patient-facing tool."
    )
    st.write(
        "Coverage Navigator forecasts whether a scheduled service may run into a preventable "
        "coverage issue before care is delivered. It finds similar past cases and patterns, "
        "flags risk, and explains the likely reasons. An insurer claims specialist verifies the "
        "case against the member's benefit documents and makes the final call."
    )

    st.subheader("Why it matters")
    s1, s2, s3 = st.columns(3)
    s1.metric("Americans carrying medical debt", "100M")
    s1.caption("Levey, 2022")
    s2.metric("Privately insured adults who correctly understand their insurance", "14%")
    s2.caption("Loewenstein et al., 2013")
    s3.metric("Denied claims that are appealed", "<1%")
    s3.caption("Claxton et al., 2025")
    st.write(
        "Today most coverage problems surface after care, when the claim is denied and the "
        "patient receives an unexpected bill. Coverage Navigator moves that check to before care."
    )

    st.subheader("Demo data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Synthetic cases", f"{len(df):,}")
    c2.metric("Synthetic issue rate", f"{df['preventable_coverage_issue'].mean():.1%}")
    c3.metric("Human final decision", "Required")

    st.subheader("Workflow")
    st.markdown(
        """
1. **Appointment scheduled** at a pilot hospital or clinic.
2. **Coverage Navigator reviews the case** in the background, inside the intake system staff already use.
3. **One of three paths:**
   - low risk → nothing changes, the appointment proceeds;
   - high risk → the case is routed to the insurer's claims specialist queue;
   - low confidence (new plan product, missing history, conflicting documentation) → **escalated** for human review instead of scored.
4. **Insurer claims specialist reviews** the flag against benefit documents, then agrees or overrides it with a reason.
5. **Specialist contacts the hospital's intake or prior-authorization team**; the hospital submits what is missing and the insurer's prior-authorization team reviews it.
6. **Coverage clarified, then care delivered**, without a surprise denial.
        """
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

        st.subheader("Data quality")
        st.caption("Situations where the model should not be trusted to score the case on its own.")
        new_product = st.checkbox("New insurance product (limited historical data)")
        missing_history = st.checkbox("Missing or outdated patient history")
        conflicting_docs = st.checkbox("Conflicting documentation (notes and codes don't align)")

        threshold = st.slider(
            "Alert threshold",
            0.10, 0.90, 0.50, 0.05,
            help="Cases at or above this risk score are routed to a specialist. "
                 "The default favors fewer, more reliable alerts so staff keep trusting them. "
                 "Lowering it catches more issues but adds many false alarms.",
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
        if not (new_product or missing_history or conflicting_docs):
            st.metric("Synthetic coverage-issue risk", f"{risk:.0%}")
        else:
            st.metric("Synthetic coverage-issue risk", "Abstained")
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

        low_conf = []
        if new_product:
            low_conf.append("New insurance product with limited historical data.")
        if missing_history:
            low_conf.append("Patient history is missing or outdated.")
        if conflicting_docs:
            low_conf.append("Clinical notes and codes do not align.")
        if low_conf:
            st.warning(
                "**Low confidence: escalate for human review.** The model abstains here "
                "instead of giving a normal score, because the data it would rely on is "
                "missing or unreliable. A senior claims specialist should review this case early."
            )
            for r in low_conf:
                st.write("• " + r)
            st.caption(f"For reference only, the raw score would have been {risk:.0%}.")
        elif risk >= threshold:
            st.error(f"Routed for review — risk is at or above the {threshold:.0%} threshold.")
        else:
            st.success(f"Below the {threshold:.0%} threshold — not routed in this scenario.")

        if reasons:
            st.write("**Contributing factors**")
            for r in reasons:
                st.write("• " + r)
        else:
            st.write("No rule-based contributing factors identified.")

        if risk >= threshold and not low_conf:
            st.info(
                "Recommended action: route to the insurer's claims specialist, who verifies the "
                "case against the member's benefit documents and contacts the hospital's intake "
                "or prior-authorization team before care."
            )

    st.caption(
        "Forecast only — not automated decision-making. A human specialist remains responsible "
        "for verification and the final determination. A case can score below the threshold "
        "and still carry a real issue: the default threshold deliberately accepts some missed "
        "cases in exchange for alerts that staff can trust."
    )


# --------------------------------------------------------------------------
# Specialist Dashboard
# --------------------------------------------------------------------------
elif page == "Specialist Dashboard":
    st.header("Specialist Dashboard")
    st.write(
        "Synthetic work queue for an **insurer claims specialist**: the cases the model routed "
        "for review before care, highest risk first."
    )

    threshold = st.slider(
        "Show cases above risk score", 0.10, 0.90, 0.50, 0.05,
        help="Default 0.50: fewer, more reliable alerts. Lower it to see how fast the queue "
             "grows and how many more alerts turn out to be false alarms.",
    )

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
        "the queue and lowers precision. When most alerts are false alarms, staff stop trusting "
        "them, so the default favors precision."
    )

    st.subheader("Review queue")
    st.caption(
        "Try it: set a status, choose Agree or Override, and give a reason for overrides. "
        "In the pilot, overrides feed back into the model and the override rate is tracked."
    )

    display = queue.head(25).copy()
    display["risk"] = display["risk"].round(2)
    display["status"] = "New"
    display["specialist_decision"] = "Not reviewed"
    display["override_reason"] = ""
    display = display[[
        "case_id", "service_type", "plan_type", "network_status", "risk", "explanation",
        "status", "specialist_decision", "override_reason",
    ]]

    edited = st.data_editor(
        display,
        width="stretch",
        hide_index=True,
        key=f"queue_{threshold:.2f}",
        disabled=["case_id", "service_type", "plan_type", "network_status", "risk", "explanation"],
        column_config={
            "risk": st.column_config.NumberColumn("Risk", format="%.2f"),
            "explanation": st.column_config.TextColumn("Why flagged", width="large"),
            "status": st.column_config.SelectboxColumn(
                "Status",
                options=["New", "Provider contacted", "Prior auth submitted", "Resolved", "Escalated"],
            ),
            "specialist_decision": st.column_config.SelectboxColumn(
                "Specialist decision", options=["Not reviewed", "Agree", "Override"],
            ),
            "override_reason": st.column_config.TextColumn("Override reason"),
        },
    )

    reviewed = edited[edited["specialist_decision"] != "Not reviewed"]
    overrides = (edited["specialist_decision"] == "Override").sum()
    missing_reason = (
        (edited["specialist_decision"] == "Override")
        & (edited["override_reason"].fillna("").str.strip() == "")
    ).sum()
    r1, r2, r3 = st.columns(3)
    r1.metric("Reviewed", f"{len(reviewed)} of {len(edited)}")
    r2.metric("Override rate", f"{overrides / len(reviewed):.0%}" if len(reviewed) else "—")
    r3.metric("Resolved before care", f"{(edited['status'] == 'Resolved').sum()}")
    if missing_reason:
        st.warning(f"{missing_reason} override(s) have no reason. Every override needs one for the audit record.")

    st.caption(
        "Decisions entered here are not saved; refreshing the page clears them. A real system "
        "would keep an auditable record of every decision, override, and escalation."
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
            "The model does not read plan rules. It learns from past claims outcomes, finds "
            "cases and patterns similar to the new one, forecasts risk, and explains the "
            "likely reasons. It never makes a coverage determination."
        )
        st.write(
            "Because it learns from claims outcomes, it can pick up billing and coding habits "
            "rather than true coverage rules. Real denial labels are also noisy: in public "
            "marketplace data, the most common denial reason is a vague \"other\", at about a "
            "third of denials. This is one reason every flag is verified by a person."
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
        st.subheader("Choosing the alert threshold")
        st.write(
            "The design favors **precision over recall**: fewer alerts, more of them real. "
            "Trust, not information, is the scarce resource. Frontline staff quickly learn to "
            "ignore a system that raises too many false alarms, and once they do, it stops "
            "preventing anything."
        )
        sweep = pd.DataFrame(metrics["threshold_sweep"])
        sweep.columns = ["Threshold", "Recall", "Precision", "Share of cases flagged"]
        st.dataframe(sweep, width="stretch", hide_index=True)
        st.caption(
            f"Held-out test set, n={metrics['n_test']}. ROC AUC {metrics['roc_auc']}. "
            "At 0.50, two out of three alerts are real issues and 14% of cases go to review. "
            "The cost is real: about three in five preventable issues are not flagged, and those "
            "missed cases may not be spread evenly across patient groups. That is why subgroup "
            "monitoring and the low-confidence escalation path are part of the design. The "
            "threshold is set and revisited by the workflow owner, not left to a default."
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

    st.subheader("Who is involved, and the incentive problem")
    st.write(
        """
        - **CMS and participating insurers** would jointly fund and govern the pilot.
        - **Insurers** employ the claims specialists and run their own prior-authorization review.
        - **Hospitals and clinics** schedule care and submit prior-authorization requests through their own intake teams.

        The hardest challenge is institutional, not technical. If the system works, it prevents
        denials that insurers would otherwise not pay, which can raise their costs. Insurers are
        likely to stay in only if fewer appeals, less rework, member satisfaction, and regulatory
        goodwill outweigh that cost. If the balance shifts, they may slow data sharing or leave
        the pilot, even if the model performs well.
        """
    )

    st.subheader("Synthetic monitoring example")
    subgroup = df.groupby("language_group")["preventable_coverage_issue"].mean()
    st.bar_chart(subgroup)
    st.caption(
        "The elevated rate in the 'Other' group is deliberately built into the synthetic data "
        "generator so that subgroup monitoring has something to detect."
    )
