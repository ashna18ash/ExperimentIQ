# ExperimentIQ Data Dictionary

All IDs are immutable strings. Timestamps are ISO-8601 UTC strings. All generated records are synthetic. The full CSV/JSON source export includes a separate `ground_truth` table/file for evaluation; it is not an analytical input table.

## 1. Entity and field definitions

| Entity | Field | Type / unit | Required | Allowed values / uniqueness | Validation rule |
|---|---|---|---|---|---|
| Experiment | `experiment_id` | string PK | Yes | `EXP-2026-NNNNNN`, unique | Must be unique and non-null |
| Experiment | `name` | string | Yes | Non-empty | Human-readable experiment name |
| Experiment | `hypothesis` | string | Yes | Predeclared statement | Must be non-empty before analysis |
| Experiment | `primary_metric` | enum | Yes | `conversion_rate` | Must match a metric definition |
| Experiment | `status` | enum | Yes | draft/running/completed/paused | Must follow lifecycle rules |
| Experiment | `start_at`, `end_at` | timestamp | Yes | UTC | `end_at >= start_at` |
| Experiment | `allocation_a`, `allocation_b` | proportion | Yes | [0,1] | Each > 0 and sum to 1 |
| Experiment | `seed` | integer | Yes | Fixed seed | Same parameters and seed reproduce the source data |
| Experiment | `analysis_plan_version` | string | Yes | Version label | Result records retain the plan version |
| Variant | `variant_id` | string PK | Yes | `VAR-<experiment-id>-A/B`, unique | Must reference one experiment |
| Variant | `experiment_id` | string FK | Yes | Existing Experiment | Parent experiment must exist |
| Variant | `name` | string | Yes | Control/Treatment | Non-empty |
| Variant | `variant_key` | enum | Yes | A/B | One control and one treatment key per experiment |
| Variant | `is_control` | boolean | Yes | true/false | A is control; B is treatment |
| AudienceMember | `user_id` | string PK | Yes | `USR-NNNNNNN`, unique | Must be unique and non-null |
| AudienceMember | `eligible` | boolean | Yes | true/false | Only eligible users may be assigned |
| AudienceMember | `segment` | enum | Yes | new/returning/high_value | Must be an allowed segment |
| AudienceMember | `country` | enum | Yes | IN/US/UK/CA | Synthetic country code |
| AudienceMember | `created_at` | timestamp | Yes | UTC | Must parse as timestamp |
| Assignment | `assignment_id` | string PK | Yes | `ASG-<experiment-id>-<user-id>`, unique | Must be unique and non-null |
| Assignment | `experiment_id` | string FK | Yes | Existing Experiment | Parent must exist |
| Assignment | `user_id` | string FK | Yes | Existing AudienceMember | User must exist and be eligible |
| Assignment | `variant_id` | string FK | Yes | Existing Variant | Variant must belong to the same experiment |
| Assignment | `assigned_at` | timestamp | Yes | UTC | Must parse; precedes valid exposure events |
| Assignment | `assignment_status` | enum | Yes | assigned | Must be an allowed lifecycle state |
| Assignment | `is_eligible` | boolean | Yes | true/false | Must be true for assigned records |
| ExposureEvent | `exposure_id` | string PK | Yes | `EXPOS-2026-NNNNNN`, unique | Must be unique and non-null |
| ExposureEvent | `assignment_id` | string FK | Yes | Existing Assignment | Parent assignment must exist |
| ExposureEvent | `experiment_id`, `user_id`, `variant_id` | string FK / denormalized references | Yes | Must match Assignment | Parent consistency check |
| ExposureEvent | `exposed_at` | timestamp | Yes | UTC | Must be >= assigned_at for valid records |
| ExposureEvent | `telemetry_status` | enum | Yes | captured/missing | Missing telemetry remains visible as an exception |
| ConversionEvent | `conversion_id` | string PK | Yes | `CONV-2026-NNNNNN`, unique | Must be unique and non-null |
| ConversionEvent | `assignment_id` | string FK | Yes | Existing Assignment | Parent assignment must exist |
| ConversionEvent | `experiment_id`, `user_id` | string FK / denormalized references | Yes | Must match Assignment | Parent consistency check |
| ConversionEvent | `converted_at` | timestamp | Yes | UTC | Within 30 days of assignment for a valid outcome |
| ConversionEvent | `conversion_value` | decimal / synthetic currency units | Yes | >= 0 | Must be non-negative |
| ConversionEvent | `event_type` | enum | Yes | purchase | Must be an allowed conversion type |
| MetricDefinition | `metric_definition_id` | string PK | Yes | `MET-NNNNNN-VN`, unique | Immutable version identifier |
| MetricDefinition | `experiment_id` | string FK | Yes | Existing Experiment | Parent must exist |
| MetricDefinition | `metric_name` | enum | Yes | conversion_rate | Must match the experiment primary metric |
| MetricDefinition | `numerator`, `denominator` | string | Yes | Documented field names | Defines the calculation explicitly |
| MetricDefinition | `unit` | enum | Yes | proportion | Values are proportions in [0,1] |
| MetricDefinition | `version` | positive integer | Yes | Monotonic per experiment | A post-result effective date is flagged as version drift |
| MetricDefinition | `effective_from`, `effective_to` | timestamp | Yes / optional | UTC / null for current | Effective intervals must be chronologically valid |
| GuardrailMetric | `guardrail_id` | string PK | Yes | Unique ID | Must be unique and non-null |
| GuardrailMetric | `experiment_id` | string FK | Yes | Existing Experiment | Parent must exist |
| GuardrailMetric | `metric_name` | enum | Yes | error_rate/refund_rate | Must be an implemented guardrail |
| GuardrailMetric | `threshold` | proportion | Yes | [0,1] | Must be within [0,1] |
| GuardrailMetric | `direction` | enum | Yes | max/min | Defines which side of threshold is a failure |
| GuardrailMetric | `unit` | enum | Yes | proportion | Must match metric unit |
| GuardrailObservation | `guardrail_observation_id` | string PK | Yes | `GRO-2026-NNNNNN`, unique | Must be unique and non-null |
| GuardrailObservation | `assignment_id` | string FK | Yes | Existing Assignment | Parent assignment must exist |
| GuardrailObservation | `experiment_id` | string FK | Yes | Existing Experiment | Must match the experiment associated with the assignment |
| GuardrailObservation | `user_id` | string FK | Yes | Existing AudienceMember | Must match the user associated with the assignment |
| GuardrailObservation | `error_occurred` | boolean | Yes | true/false | Must be a valid boolean outcome |
| GuardrailObservation | `refund_occurred` | boolean | Yes | true/false | Must be a valid boolean outcome |
| GuardrailObservation | `observed_at` | timestamp | Yes | UTC | Must parse as a valid timestamp and satisfy observation chronology checks |
| ExperimentResult | `experiment_result_id` | string PK | Yes | `RES-<experiment-id>-V1` or subsequent version | Unique result snapshot identifier |
| ExperimentResult | `experiment_id` | string FK | Yes | Existing Experiment | Parent must exist |
| ExperimentResult | `analysis_plan_version`, `generated_at` | string / timestamp | Yes | Version / UTC | Preserves analysis plan and result timestamp |
| ExperimentResult | `dataset_seed`, `dataset_size` | integer / row count | Yes | Generator seed and audience row count | Identifies the synthetic source run used for the snapshot |
| ExperimentResult | arm counts and conversions | integer counts | Yes | >= 0 | Reconcile to assignment and conversion source records |
| ExperimentResult | rates, uplift, p-value, CI | proportion / percent | Yes | p-value in [0,1] | Derived from source records, not independently randomized |
| ExperimentResult | `srm_status`, `decision` | enum/string | Yes | PASS/FAIL and decision labels | Derived from allocation, evidence and guardrails |

## 2. Relationships and cardinality

- Experiment 1:N Variant.
- Experiment 1:N Assignment.
- AudienceMember 1:N Assignment across experiment history.
- Variant 1:N Assignment.
- Assignment 1:N ExposureEvent (optional; noncompliance means zero exposures).
- Assignment 1:N ConversionEvent (optional; zero or more conversion events).
- Experiment 1:N MetricDefinition (effective-dated version history).
- Experiment 1:N GuardrailMetric.
- Experiment 1:N GuardrailObservation.
- Assignment 1:N GuardrailObservation.
- AudienceMember 1:N GuardrailObservation.
- Experiment 1:N ExperimentResult (versioned snapshots; unchanged regenerations are idempotent and changed datasets append a new result version).

Assignment is the bridge linking AudienceMember and Variant within an Experiment. GuardrailObservation records are associated with their corresponding Assignment, Experiment and AudienceMember. No unmodelled many-to-many relationship is required by the current scope.

The current generator creates one GuardrailObservation for each normal assignment. Each observation stores an error outcome, refund outcome and observation timestamp.

## 3. Lifecycle and exception handling

- Experiment: draft -> running -> completed, or paused.
- Assignment: assigned -> zero or more exposure events -> zero or more conversion events.
- GuardrailObservation records capture assignment-level error and refund outcomes. They retain references to the related assignment, experiment and audience member and are validated for referential consistency.
- Event and observation records are retained as source records. Deliberate negative scenarios are labelled in the separate `ground_truth` evaluation file and detected in the exception queue.
- Source entity/event tables are regenerated as an explicit local demo reset. Experiment result snapshots are preserved and versioned across changed regenerations; identical reruns do not create duplicate snapshots. A production implementation should use append-only event history and database migrations rather than replacing source tables.

## 4. Ground-truth scenario labels

The following scenario labels are generated from documented injection rules:

- `MULTIPLE_VARIANT_ASSIGNMENT`
- `EXPOSURE_BEFORE_ASSIGNMENT`
- `OUTCOME_WINDOW_INCONSISTENT`
- `METRIC_CHANGED_AFTER_RESULTS`
- `SAMPLE_RATIO_MISMATCH`

They are stored separately from analytical input data and used only to evaluate whether the validation layer detects the known scenario types.