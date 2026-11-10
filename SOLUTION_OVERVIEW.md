# ExperimentIQ — Solution Overview

## 1. Problem statement in plain English

Organizations run experiments to decide whether a proposed product, marketing or customer-experience change should be adopted. A valid experiment must define what success means before it starts, assign eligible users according to a declared randomization plan, capture exposure and outcome events, monitor operational guardrails, and estimate the treatment effect with uncertainty. ExperimentIQ demonstrates this complete workflow and makes exceptions visible so that a decision is traceable rather than based on a conversion percentage alone.

## 2. Objective and scope

**Objective:** Design experiments, validate randomization, calculate guardrail metrics and estimate treatment effects.

**In scope:** Synthetic experiment design and master data; audience eligibility; fixed-seed assignment; exposure and conversion event generation; noncompliance and incomplete telemetry; predeclared metric definitions; guardrail thresholds and observations; randomization and integrity checks; treatment-effect analysis; result snapshots; exception evaluation; relational persistence; exports and reproducibility.

**Out of scope:** Live customer data, production feature-flag integration, actual campaign execution, personally identifiable information, authentication/authorization and production-grade causal inference.

## 3. Primary actors and what they do

| Actor | Role in the workflow |
|---|---|
| Customer / prospect | Receives the control or treatment experience; may or may not have an exposure event recorded. |
| Marketing / sales operations | Defines the experiment hypothesis, audience, variants, primary metric and guardrail plan. |
| Support agent | Represents the support workflow and helps surface experience or service issues reflected by guardrails. |
| Analyst | Reviews assignment quality, event data, exceptions, statistical uncertainty and rollout evidence. |
| System administrator | Maintains the synthetic environment, database, configuration and export process. |

## 4. End-to-end workflow

1. **Define the experiment.** Create the experiment, hypothesis, variants, primary metric, guardrails, allocation ratio, lifecycle and analysis-plan version.
2. **Generate master data.** Create experiments, variants and eligible audience members first, with stable identifiers.
3. **Randomize.** Assign each eligible audience member to a variant using a fixed seed and the experiment's declared allocation.
4. **Generate dependent events.** Create exposure and conversion records using valid parent IDs. Generate assignment-level guardrail observations. Simulate noncompliance, missing telemetry, delayed conversions and controlled treatment effects.
5. **Validate evidence.** Check primary and foreign keys, parent consistency, eligibility, event ordering, outcome windows, configured allocation, metric-version timing, enums, guardrail definitions and guardrail observation relationships.
6. **Calculate outcomes.** Apply the predeclared intention-to-treat metric, estimate uplift and uncertainty, test sample-ratio mismatch against the configured allocation, and evaluate guardrails using recorded error and refund observations.
7. **Record decisions.** Persist result snapshots with a result ID, analysis-plan version, source seed and timestamp. Changed dataset runs append a new result version; identical reruns are idempotent.
8. **Evaluate and export.** Compare detected issue categories with separately stored ground truth, export CSV/JSON/SQL and documentation, and verify fixed-seed checksums.

## 5. Data model and relationships

| Entity | Primary key | Main relationships |
|---|---|---|
| Experiment | `experiment_id` | Parent of Variant, Assignment, MetricDefinition, GuardrailMetric, GuardrailObservation and ExperimentResult |
| Variant | `variant_id` | Belongs to one Experiment; referenced by Assignment |
| AudienceMember | `user_id` | May have Assignment records across multiple experiments |
| Assignment | `assignment_id` | References one Experiment, AudienceMember and Variant; parent of ExposureEvent, ConversionEvent and GuardrailObservation |
| ExposureEvent | `exposure_id` | References one Assignment and retains experiment/user/variant references for consistency validation |
| ConversionEvent | `conversion_id` | References one Assignment and retains experiment/user references for consistency validation |
| MetricDefinition | `metric_definition_id` | Versioned definition belonging to one Experiment |
| GuardrailMetric | `guardrail_id` | Threshold definition belonging to one Experiment |
| GuardrailObservation | `guardrail_observation_id` | References one Assignment, Experiment and AudienceMember; stores observed error and refund outcomes |
| ExperimentResult | `experiment_result_id` | Immutable analysis snapshot belonging to one Experiment |

**Cardinality and relationships:**

- Experiment 1:N Variant.
- Experiment 1:N Assignment.
- AudienceMember 1:N Assignment across experiment history.
- Variant 1:N Assignment.
- Assignment 1:N optional ExposureEvent.
- Assignment 1:N optional ConversionEvent.
- Experiment 1:N MetricDefinition.
- Experiment 1:N GuardrailMetric.
- Experiment 1:N GuardrailObservation.
- Assignment 1:N GuardrailObservation.
- AudienceMember 1:N GuardrailObservation.
- Experiment 1:N ExperimentResult.

Assignment bridges AudienceMember and Variant in the context of an Experiment.

The current generator creates one GuardrailObservation for each normal assignment. Each observation records the related assignment, experiment and user, along with boolean error and refund outcomes and an observation timestamp.

## 6. Analysis rules

- **Primary metric:** Unique assigned users with at least one valid conversion / assigned users.
- **Attribution:** Intention-to-treat; users remain in their originally assigned group whether or not exposure telemetry exists.
- **Valid conversion window:** From assignment time through 30 days after assignment, inclusive.
- **Treatment effect:** Treatment conversion rate minus control conversion rate; relative uplift is also displayed.
- **Uncertainty:** Wilson confidence intervals per arm and an approximate two-sided two-proportion z-test / normal confidence interval for the difference.
- **Randomization:** Chi-square sample-ratio mismatch test against the configured A/B allocation, not an assumed 50:50 split.
- **Guardrails:** Recorded synthetic error and refund outcomes are aggregated into rates and evaluated against configured maximum thresholds.
- **Decision labels:** Decision support, not automatic production rollout commands.

## 7. Negative scenarios and expected detection

| Scenario | Generator action | Validation expectation |
|---|---|---|
| User assigned to multiple variants | Adds a second assignment for the same user and experiment with the other variant | `MULTIPLE_VARIANT_ASSIGNMENT` |
| Exposure before assignment | Adds an exposure timestamp earlier than its assignment | `EXPOSURE_BEFORE_ASSIGNMENT` |
| Outcome window inconsistent | Adds a conversion 60 days after assignment | `OUTCOME_WINDOW_INCONSISTENT`; excluded from primary conversion metric |
| Sample-ratio mismatch | Changes a labelled subset of assignments in one experiment so observed counts diverge from the declared split | `SAMPLE_RATIO_MISMATCH` |
| Metric changed after results | Adds a later effective metric-definition version after the first stored result timestamp | `METRIC_CHANGED_AFTER_RESULTS` |

The expected labels are stored in a separate `ground_truth` table and export. They are used only for category-level evaluation and are not merged into analytical features.

## 8. Acceptance evidence

- Fixed-seed generation creates 50,000 audience rows across 10 experiments by default.
- All five deliberate anomaly categories are detected in the full-size validation run.
- SQLite tables declare primary keys and foreign keys; data integrity checks cover key uniqueness and joins.
- `experiment_results` contains derived result records rather than an empty placeholder.
- Results expose effect size, uncertainty, randomization status and guardrail outcomes.
- Assignment-level guardrail observations are persisted and validated for parent references, consistency, boolean outcomes and timestamps.
- CSV, JSON and SQL exports are generated with documentation and a SHA-256 manifest.
- The SQL export imports successfully into a fresh SQLite database, with integrity check `ok` and no foreign-key violations.
- Automated tests cover deterministic generation, anomaly detection, SRM, foreign-key enforcement, result persistence and outcome-window filtering.

## 9. Limitations and next steps

The dataset is synthetic. Guardrail signals are simulated. Statistical tests are intentionally transparent approximations for a hackathon prototype and should be reviewed before production use.

The app resets source data only when the user explicitly generates a new dataset; result snapshots are retained and versioned. Identical reruns with the same generation parameters are idempotent.

A production version should use append-only source events, schema migrations, access controls, audit logging, background jobs, monitoring and a reviewed statistical analysis plan.