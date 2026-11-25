# Governance & Responsible AI

Coverage Navigator is intentionally designed as a **decision-support system**, not an automated coverage-decision engine.

## Governance Questions
- Who is accountable for the final decision?
- What data may the model access?
- How are payer-specific labels standardized?
- How are overrides recorded?
- How often is subgroup performance reviewed?
- When should the system abstain?
- What happens when organizations disagree about incentives or data sharing?

## Answers the Design Commits To
- **Accountability:** the insurer's claims specialist makes every final coverage decision. The model never does.
- **Abstention:** the model abstains and escalates to a human when the case involves a new insurance product with limited history, missing or outdated patient history, or conflicting documentation. Provider disagreement is also escalated, because clinical context matters.
- **Overrides:** every override is recorded with a reason. The override rate is tracked during the pilot and fed back into model updates.
- **Threshold:** set by the workflow owner to favor precision over recall, so alerts stay trustworthy. It is reviewed at each pilot phase.

## Questions Still Open for the Pilot
These come from the frontline staff briefing and are meant to be answered with staff:
- Which alerts are most useful?
- Where does the workflow slow down?
- How should confidence scores be displayed?
- Which cases require immediate escalation?
- What feedback should improve future versions?

## Privacy
A real deployment would require minimum-necessary data use, role-based access, encryption, auditable access logs, retention limits, formal data-sharing agreements, and privacy/security review before expansion.

## Institutional Incentives
The intervention creates a genuine policy tension: preventing denials may improve patient experience while increasing paid claims. Successful implementation therefore depends on aligning incentives across public and private actors—not just building a technically strong model.
