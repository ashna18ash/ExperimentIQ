# ExperimentIQ — Data Dictionary
### Observed values versus supported values

Allowed values documented in this file distinguish between:

- **Observed values:** Values actually present in the current generated SQLite dataset.
- **Supported values:** Values accepted or handled by application-level business logic.

The observed dataset is not necessarily an exhaustive representation of every supported application state. For example, the current dataset contains only eligible audience members (`eligible = 1`), while the schema and application logic represent eligibility as a Boolean field.

SQLite declares primary-key and foreign-key constraints, but does not declare explicit `NOT NULL`, `UNIQUE`, or `CHECK` constraints on the documented columns. Required fields and business validation are therefore enforced through application logic and data-generation rules.

**Version:** 1.1
**Project:** Experimentation & Decision Intelligence
**Database:** SQLite
**Default dataset:** 50,000 audience members, 10 experiments
**Default random seed:** `20261022`
**Timestamp convention:** ISO 8601 text timestamps, generated in UTC.

---

## 1. Documentation conventions

### Required-field notation

- **Yes*** — required and populated by the current dataset generator; SQLite does not necessarily enforce `NOT NULL`.
- **No** — may be `NULL` in the generated dataset.
- **PK** — primary key declared in SQLite; uniqueness is enforced, while non-null values are additionally checked by application validation.
- **FK** — foreign key, enforced by SQLite.
- **Unique** — uniqueness requirement. A field is not necessarily protected by a separate SQLite `UNIQUE` constraint unless identified as a primary key.

### SQLite data types

| Type | Meaning |
|---|---|
| TEXT | Identifiers, descriptive values, categories and ISO timestamps |
| INTEGER | Whole numbers and Boolean values (`0` or `1`) |
| REAL | Decimal numbers, proportions, rates, monetary values and statistical measurements |

### Validation conventions

- Identifier values must follow their documented formats.
- Foreign keys must reference an existing parent record.
- Timestamps must be parseable ISO 8601 values.
- Allocation proportions must be between 0 and 1 and sum to 1.
- Proportions and rates must be between 0 and 1 unless explicitly documented as percentages.
- Event chronology and experiment outcome-window rules are checked by application-level validation.
- Ground-truth labels are kept separate from analytical inputs.

**Important implementation note:** The current SQLite schema declares primary keys and foreign keys. Other field requirements, allowed values, and business rules are application-level validation requirements rather than universal SQLite `NOT NULL`, `CHECK`, or `UNIQUE` constraints.

---

# 2. Entity relationship overview

| Parent entity | Child entity | Relationship | Relationship rule |
|---|---|---|---|
| Experiment | Variant | 1:N | Each variant belongs to one experiment |
| Experiment | Assignment | 1:N | Each assignment belongs to one experiment |
| AudienceMember | Assignment | 1:N | A user may participate in different experiments |
| Variant | Assignment | 1:N | Each assignment references one variant |
| Assignment | ExposureEvent | 1:N | An assignment may have zero or more exposures |
| Assignment | ConversionEvent | 1:N | An assignment may have zero or more conversions |
| Experiment | MetricDefinition | 1:N | Definitions may have multiple versions |
| Experiment | GuardrailMetric | 1:N | An experiment can have multiple guardrails |
| Assignment | GuardrailObservation | 1:N | Observations are associated with an assignment |
| Experiment | ExperimentResult | 1:N | Result snapshots are retained across changed runs |

`ground_truth` is an independent evaluation-label table. It does not have declared foreign keys or a declared primary key in the current implementation.

---

# 3. Field-level data dictionary

## 3.1 experiments

**Purpose:** Stores experiment master records, hypotheses, allocation configuration, lifecycle information and reproducibility settings.

**Primary key:** `experiment_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| experiment_id | TEXT | Yes* | `EXP-2026-000001` format | Yes — PK | Must be a valid experiment identifier |
| name | TEXT | Yes* | Experiment descriptive name | No | Must not be empty |
| hypothesis | TEXT | Yes* | Free-text hypothesis | No | Must describe the intended treatment effect |
| primary_metric | TEXT | Yes* | `conversion_rate` in current dataset | No | Must match a defined metric |
| status | TEXT | Yes* | `completed` in current dataset | No | Must be a recognized lifecycle state |
| start_at | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must parse as a valid timestamp |
| end_at | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must be later than or equal to start_at |
| allocation_a | REAL | Yes* | Proportion from 0 to 1 | No | Must be between 0 and 1 |
| allocation_b | REAL | Yes* | Proportion from 0 to 1 | No | Must be between 0 and 1; A + B must equal 1 |
| seed | INTEGER | Yes* | Integer random seed; default `20261022` | No | Must be an integer used for reproducible generation |
| analysis_plan_version | TEXT | Yes* | Version string; currently `1.0` | No | Must identify the analysis-plan version |

## 3.2 variants

**Purpose:** Defines the control and treatment arms belonging to each experiment.

**Primary key:** `variant_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| variant_id | TEXT | Yes* | `VAR-<experiment-id>-A/B` | Yes — PK | Must identify a valid experiment variant |
| experiment_id | TEXT | Yes* | Existing experiment identifier | No | FK → `experiments.experiment_id` |
| name | TEXT | Yes* | `Control`, `Treatment` | No | Must match the variant key's intended role |
| variant_key | TEXT | Yes* | `A`, `B` | No | Must be one of the supported experiment arms |
| is_control | INTEGER | Yes* | Boolean: `0` or `1` | No | A must be control (`1`); B must be treatment (`0`) |

**Relationship:** One experiment has multiple variants. The current generator creates two variants per experiment.

## 3.3 audience_members

**Purpose:** Stores synthetic user-level audience master data and eligibility information.

**Primary key:** `user_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| user_id | TEXT | Yes* | `USR-0000001` format | Yes — PK | Must be a valid user identifier |
| eligible | INTEGER | Yes* | Boolean: `0` or `1` | No | Must represent audience eligibility |
| segment | TEXT | Yes* | `new`, `returning`, `high_value` | No | Must belong to the supported segment categories |
| country | TEXT | Yes* | `IN`, `US`, `UK`, `CA` | No | Must be a supported country code in the dataset |
| created_at | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must parse as a valid timestamp |

**Relationship:** One audience member may have assignments in multiple experiments. The current generated dataset distributes audience members across experiments.

## 3.4 assignments

**Purpose:** Records randomized assignment of eligible users to experiment variants. This is the source of truth for intention-to-treat analysis.

**Primary key:** `assignment_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| assignment_id | TEXT | Yes* | `ASG-<experiment-id>-<user-id>` | Yes — PK | Must identify an assignment record |
| experiment_id | TEXT | Yes* | Existing experiment identifier | No | FK → `experiments.experiment_id` |
| user_id | TEXT | Yes* | Existing user identifier | No | FK → `audience_members.user_id` |
| variant_id | TEXT | Yes* | Existing variant identifier | No | FK → `variants.variant_id`; variant must belong to the same experiment |
| assigned_at | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must parse as a valid assignment timestamp |
| assignment_status | TEXT | Yes* | `assigned` in current dataset | No | Must be a recognized assignment state |
| is_eligible | INTEGER | Yes* | Boolean: `0` or `1` | No | Must agree with eligibility rules for the assigned user |

**Business rule:** A user should not be assigned to conflicting variants within the same experiment. The validation layer deliberately identifies the conflicting-assignment anomaly.

## 3.5 exposure_events

**Purpose:** Records whether an assigned user was exposed to the assigned experiment variant and the state of exposure telemetry.

**Primary key:** `exposure_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| exposure_id | TEXT | Yes* | `EXPOS-2026-000001` format | Yes — PK | Must be a valid exposure identifier |
| assignment_id | TEXT | Yes* | Existing assignment identifier | No | FK → `assignments.assignment_id` |
| experiment_id | TEXT | Yes* | Existing experiment identifier | No | FK → `experiments.experiment_id`; must match assignment |
| user_id | TEXT | Yes* | Existing user identifier | No | FK → `audience_members.user_id`; must match assignment |
| variant_id | TEXT | Yes* | Existing variant identifier | No | FK → `variants.variant_id`; must match assignment |
| exposed_at | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must be at or after assignment time |
| telemetry_status | TEXT | Yes* | `captured`, `missing` | No | Must be one of the supported telemetry states |

**Business rule:** Exposure is not assumed merely because an assignment exists. Missing exposure and telemetry records remain distinguishable.

## 3.6 conversion_events

**Purpose:** Stores conversion outcomes attributed to an assignment, including conversion time, value and event category.

**Primary key:** `conversion_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| conversion_id | TEXT | Yes* | `CONV-2026-000001` format | Yes — PK | Must be a valid conversion identifier |
| assignment_id | TEXT | Yes* | Existing assignment identifier | No | FK → `assignments.assignment_id` |
| experiment_id | TEXT | Yes* | Existing experiment identifier | No | FK → `experiments.experiment_id`; must match assignment |
| user_id | TEXT | Yes* | Existing user identifier | No | FK → `audience_members.user_id`; must match assignment |
| converted_at | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must be at or after assignment time and within the 30-day analysis window to count |
| conversion_value | REAL | Yes* | Positive numeric value; synthetic currency-like units | No | Must be numeric and non-negative |
| event_type | TEXT | Yes* | `purchase` in current dataset | No | Must be a recognized conversion event type |

**Business rule:** The primary conversion analysis counts an assignment at most once, even if multiple conversion records exist. Conversion events outside the 30-day window are excluded from the primary metric.

## 3.7 metric_definitions

**Purpose:** Stores versioned definitions for experiment metrics, including numerator, denominator and effective dates.

**Primary key:** `metric_definition_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| metric_definition_id | TEXT | Yes* | `MET-000001-V1` format | Yes — PK | Must identify a metric-definition version |
| experiment_id | TEXT | Yes* | Existing experiment identifier | No | FK → `experiments.experiment_id` |
| metric_name | TEXT | Yes* | `conversion_rate` in current dataset | No | Must identify a recognized metric |
| numerator | TEXT | Yes* | `converted_assigned_users` or `converted_exposed_users` in current scenarios | No | Must describe the numerator population |
| denominator | TEXT | Yes* | `assigned_users` or `exposed_users` in current scenarios | No | Must describe the denominator population |
| unit | TEXT | Yes* | `proportion` in current dataset | No | Must identify the measurement unit |
| version | INTEGER | Yes* | Positive integer; versions 1, 2, etc. | No | Must be greater than zero and identify the definition version |
| effective_from | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must parse as a valid effective-start timestamp |
| effective_to | TEXT | No | ISO 8601 UTC timestamp or NULL | No | When present, must be later than or equal to effective_from |

**Business rule:** The analytics layer selects the latest metric definition effective at the analysis timestamp. A definition introduced after the analysis snapshot must not silently change that historical result.

## 3.8 guardrail_metrics

**Purpose:** Defines experiment safety metrics and their acceptable thresholds.

**Primary key:** `guardrail_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| guardrail_id | TEXT | Yes* | `GRD-000001-01` format | Yes — PK | Must identify a guardrail definition |
| experiment_id | TEXT | Yes* | Existing experiment identifier | No | FK → `experiments.experiment_id` |
| metric_name | TEXT | Yes* | `error_rate`, `refund_rate` | No | Must be a supported guardrail metric |
| threshold | REAL | Yes* | Proportion from 0 to 1 | No | Must be within the valid proportion range |
| direction | TEXT | Yes* | `max`, `min` | No | `max` means the observed rate must not exceed the threshold; `min` means it must not fall below it |
| unit | TEXT | Yes* | `proportion` | No | Must match the metric's measurement unit |

**Current thresholds:** Error rate 0.08; refund rate 0.06. These are synthetic demonstration thresholds.

## 3.9 guardrail_observations

**Purpose:** Stores assignment-level source observations used to calculate actual control and treatment guardrail rates.

**Primary key:** `guardrail_observation_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| guardrail_observation_id | TEXT | Yes* | `GRO-2026-000001` format | Yes — PK | Must identify a unique observation |
| assignment_id | TEXT | Yes* | Existing assignment identifier | No | FK → `assignments.assignment_id` |
| experiment_id | TEXT | Yes* | Existing experiment identifier | No | FK → `experiments.experiment_id`; must match assignment |
| user_id | TEXT | Yes* | Existing user identifier | No | FK → `audience_members.user_id`; must match assignment |
| error_occurred | INTEGER | Yes* | Boolean: `0` or `1` | No | Must represent whether an error occurred |
| refund_occurred | INTEGER | Yes* | Boolean: `0` or `1` | No | Must represent whether a refund occurred |
| observed_at | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must parse as a valid observation timestamp and follow assignment time |

**Business rule:** Guardrail rates are calculated from these source observations and the corresponding assignment variants. They are not independently invented by the analytics layer.

## 3.10 experiment_results

**Purpose:** Stores persisted experiment-analysis snapshots, including sample sizes, conversions, rates, uncertainty, SRM status and the resulting decision.

**Primary key:** `experiment_result_id`

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| experiment_result_id | TEXT | Yes* | `RES-<experiment-id>-V1` format | Yes — PK | Must identify a unique result snapshot version |
| experiment_id | TEXT | Yes* | Existing experiment identifier | No | FK → `experiments.experiment_id` |
| analysis_plan_version | TEXT | Yes* | Version string; currently `1.0` | No | Must identify the analysis plan used |
| generated_at | TEXT | Yes* | ISO 8601 UTC timestamp | No | Must parse as a valid analysis snapshot timestamp |
| dataset_seed | INTEGER | Yes* | Integer random seed | No | Must match the seed used to generate the source dataset |
| dataset_size | INTEGER | Yes* | Non-negative count of audience records | No | Must be consistent with the generated dataset |
| control_assigned | INTEGER | Yes* | Non-negative assignment count | No | Must be consistent with the analyzed control arm |
| treatment_assigned | INTEGER | Yes* | Non-negative assignment count | No | Must be consistent with the analyzed treatment arm |
| control_conversions | INTEGER | Yes* | Non-negative conversion count | No | Must not exceed control_assigned |
| treatment_conversions | INTEGER | Yes* | Non-negative conversion count | No | Must not exceed treatment_assigned |
| control_rate | REAL | Yes* | Proportion from 0 to 1 | No | Must equal control_conversions divided by control_assigned when the denominator is positive |
| treatment_rate | REAL | Yes* | Proportion from 0 to 1 | No | Must equal treatment_conversions divided by treatment_assigned when the denominator is positive |
| uplift_pct | REAL | Yes* | Relative percentage uplift; percentage points are not implied | No | Calculated from treatment and control rates; must be finite |
| p_value | REAL | Yes* | Probability from 0 to 1 | No | Must be a valid statistical p-value |
| ci_low | REAL | Yes* | Absolute difference in conversion proportions | No | Must be the lower bound of the treatment-minus-control confidence interval |
| ci_high | REAL | Yes* | Absolute difference in conversion proportions | No | Must be greater than or equal to ci_low |
| srm_status | TEXT | Yes* | `PASS`, `FAIL` | No | Must reflect the sample-ratio mismatch assessment against declared allocation |
| decision | TEXT | Yes* | `DO NOT ROLL OUT - SAMPLE RATIO MISMATCH`, `DO NOT ROLL OUT - GUARDRAIL FAILURE`, `PROMISING - REVIEW`, `TREATMENT UNDERPERFORMS`, `INCONCLUSIVE` | No | Must follow the documented decision rules |

**Snapshot behavior:** Initial results use version `V1`. When a changed dataset produces a different result, a new version is appended. Identical results are not duplicated. Result snapshots are retained when source data is regenerated.

---

# 4. Evaluation labels

## 4.1 ground_truth

**Purpose:** Stores expected anomaly labels for evaluating whether validation and decision logic identify deliberately injected data-quality scenarios.

**Primary key:** None declared.
**Foreign keys:** None declared.

| Field | SQLite type | Required | Allowed values / units | Unique | Validation |
|---|---|---|---|---|---|
| entity_type | TEXT | Yes* | `Assignment`, `ExposureEvent`, `ConversionEvent`, `MetricDefinition`, `Experiment` | No | Must identify the entity containing the injected anomaly |
| entity_id | TEXT | Yes* | Identifier of the affected entity | No | Must refer to the intended generated anomaly record |
| issue_type | TEXT | Yes* | `MULTIPLE_VARIANT_ASSIGNMENT`, `EXPOSURE_BEFORE_ASSIGNMENT`, `OUTCOME_WINDOW_INCONSISTENT`, `METRIC_CHANGED_AFTER_RESULTS`, `SAMPLE_RATIO_MISMATCH` | No | Must identify a supported anomaly category |
| expected_decision | TEXT | Yes* | `FLAG_CONFLICTING_ASSIGNMENT`, `FLAG_INVALID_EVENT_ORDER`, `FLAG_OUTSIDE_WINDOW`, `FLAG_VERSION_DRIFT`, `FLAG_ALLOCATION_MISMATCH` | No | Must match the documented expected response for the issue type |

**Evaluation rule:** Ground-truth records must not be used as input to the analytical decision calculation. They exist separately so the application can compare detected issues with known synthetic anomalies.

---

# 5. Defined anomaly scenarios

| Scenario | Affected entity | Issue type | Expected decision |
|---|---|---|---|
| Same user assigned to conflicting variants | Assignment | `MULTIPLE_VARIANT_ASSIGNMENT` | `FLAG_CONFLICTING_ASSIGNMENT` |
| Exposure timestamp precedes assignment | ExposureEvent | `EXPOSURE_BEFORE_ASSIGNMENT` | `FLAG_INVALID_EVENT_ORDER` |
| Conversion occurs outside the 30-day outcome window | ConversionEvent | `OUTCOME_WINDOW_INCONSISTENT` | `FLAG_OUTSIDE_WINDOW` |
| Metric definition changes after analysis snapshot | MetricDefinition | `METRIC_CHANGED_AFTER_RESULTS` | `FLAG_VERSION_DRIFT` |
| Observed assignment distribution violates declared allocation | Experiment | `SAMPLE_RATIO_MISMATCH` | `FLAG_ALLOCATION_MISMATCH` |

---

# 6. Integrity and business validation rules

The following rules complement the field-level definitions.

1. **Primary-key integrity:** Every declared primary key must be unique; application validation also checks that primary-key values are non-null.
2. **Foreign-key integrity:** Every child reference must resolve to an existing parent record.
3. **Variant consistency:** An assignment's variant must belong to the same experiment as the assignment.
4. **Assignment consistency:** A user must not receive conflicting variants within the same experiment.
5. **Eligibility:** Only eligible users should be assigned.
6. **Exposure chronology:** Exposure must not precede assignment.
7. **Conversion attribution:** A conversion must reference the correct assignment, user and experiment.
8. **Outcome window:** Only conversions between assignment time and 30 days after assignment count toward the primary conversion metric.
9. **Metric versioning:** Metric definitions must be selected according to their effective dates.
10. **Randomization quality:** Observed arm counts must be checked against the experiment's declared allocation.
11. **Guardrail assessment:** Observed guardrail rates must be compared with the configured thresholds.
12. **Result traceability:** Result snapshots must preserve the analysis-plan version, generation timestamp, dataset seed and dataset size.
13. **Ground-truth separation:** Evaluation labels must not influence the analytical results being evaluated.

---

# 7. Current dataset reference

| Dataset characteristic | Current default |
|---|---:|
| Experiments | 10 |
| Audience members | 50,000 |
| Variants | 20 |
| Assignments | 50,001, including one deliberate conflicting-assignment anomaly |
| Guardrail observations | 50,000 |
| Guardrail definitions | 20 |
| Metric definitions | 11 |
| Ground-truth anomaly labels | 5 |
| Analysis outcome window | 30 days |
| Default seed | 20261022 |

The dataset is synthetic and intended for experimentation-system validation, reproducibility testing and demonstration. It does not represent actual customer activity.

---

# 8. Schema maintenance

When a field is added, removed or renamed in the generator or SQLite storage layer:

1. Update this dictionary.
2. Verify the generated DataFrame column.
3. Verify the SQLite column and its declared type.
4. Update primary-key or foreign-key documentation if relationships change.
5. Update validation rules and tests.
6. Regenerate and verify the exported documentation bundle.

**Source of truth:** `src/generator.py`, `src/storage.py`, `src/validation.py` and `src/analytics.py`.
