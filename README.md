# ExperimentIQ — Experimentation & Decision Intelligence

ExperimentIQ is a Streamlit hackathon prototype that demonstrates an end-to-end experimentation workflow: define an experiment, create a reproducible eligible audience, randomize assignments, record exposures and conversions, validate data quality, calculate operational guardrails, and estimate treatment effects with uncertainty.

## Business problem

A team cannot safely decide whether to roll out a new experience by comparing two conversion percentages alone. It must also know whether users were assigned according to the experiment plan, whether event records are valid, whether outcomes fall inside the analysis window, whether operational guardrails regressed, and how uncertain the estimated effect is.

ExperimentIQ brings these checks into one traceable workflow and exposes exceptions rather than hiding them.

## Primary actors

- **Customer / prospect:** Receives the control or treatment experience.
- **Marketing / sales operations:** Defines the hypothesis, audience, variants, and launch plan.
- **Support agent:** Supports users and surfaces experience issues.
- **Analyst:** Validates evidence, evaluates treatment effects, and recommends next steps.
- **System administrator:** Maintains data, configuration, and system operation.

## Main capabilities

- Experiment registry with hypothesis, primary metric, lifecycle, and planned allocation.
- Deterministic synthetic dataset generator (default 50,000 audience members across 10 experiments; seed `20261022`).
- Fixed-seed random assignment with balanced and unbalanced allocation plans.
- Exposure and conversion event simulation, noncompliance, and missing telemetry.
- Assignment-level guardrail observations recording operational error and refund outcomes.
- Intention-to-treat conversion analysis, relative uplift, p-value, and 95% confidence interval.
- Configured-allocation sample-ratio mismatch (SRM) diagnostics; no hard-coded 50:50 assumption.
- Guardrail monitoring with visible PASS / FAIL states based on recorded synthetic outcomes and configured thresholds.
- Exception queue for duplicate variant assignments, exposure-before-assignment, conversions outside the 30-day outcome window, post-result metric version changes, sample-ratio mismatch, and missing telemetry.
- Persisted `ExperimentResult` snapshots derived from source records.
- Separate `ground_truth` table/file for labelled scenarios; it is not joined into analytical inputs.
- Primary-key, foreign-key, parent consistency, lifecycle, timestamp, enum, and guardrail-definition checks.
- CSV, JSON, and SQL seed exports plus README, data dictionary, and SHA-256 source manifest.
- SQLite primary-key and foreign-key constraints; result snapshots are versioned and retained across changed regenerations.

## Run on Windows PowerShell

The project uses Python 3.10.11.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

Select **Generate dataset** in the sidebar to create or reset the local demo dataset. You can select a smaller audience size while developing and use 50,000 for the full demonstration.

If you already have the project virtual environment configured, activate it rather than recreating it.

## Run tests

From the project directory:

```powershell
python -m pytest -v
```

The test suite covers:

- Deterministic generation.
- All five labelled anomaly scenarios.
- Result snapshot persistence and history.
- SQLite relational schema and foreign-key integrity.
- Uncertainty outputs.
- SRM detection against configured allocation.
- Primary metric outcome-window handling.

Latest verified test result: **14 tests passed**.

## Reproducibility

```powershell
python -c "from src.generator import generate_dataset; f=generate_dataset(50000,10,20261022); print({k:len(v) for k,v in f.items()})"
```

Run the command twice with identical arguments to regenerate identical source records.

The default configuration uses:

- Audience size: `50000`
- Experiment count: `10`
- Random seed: `20261022`

The application stores its local SQLite database in `data/experimentiq.db`.

The **Data & Exports** page creates CSV, JSON, and SQL exports, includes the README and data dictionary, and writes a SHA-256 manifest.

Result snapshots are retained as versioned records when a changed dataset is generated. Repeating an identical dataset does not create duplicate snapshots.

The `ground_truth` table is exported separately and must not be merged into analytical input data.

## Data model and relationships

The relational model includes:

- **Experiment 1:N Variant**
- **Experiment 1:N Assignment**
- **AudienceMember 1:N Assignment** across experiment history.
- **Variant 1:N Assignment**
- **Assignment 1:N ExposureEvent** (optional; no exposure represents noncompliance).
- **Assignment 1:N ConversionEvent** (optional; zero or more outcomes).
- **Experiment 1:N MetricDefinition** (version history).
- **Experiment 1:N GuardrailMetric**
- **Assignment 1:N GuardrailObservation**
- **Experiment 1:N GuardrailObservation**
- **AudienceMember 1:N GuardrailObservation**
- **Experiment 1:N ExperimentResult** (versioned immutable result snapshots).

`Assignment` is the bridge between `AudienceMember` and `Variant` within an `Experiment`.

`GuardrailObservation` records assignment-level operational outcomes, including whether an error or refund occurred. Its `assignment_id`, `experiment_id`, and `user_id` columns reference the corresponding parent records through SQLite foreign keys.

See `SOLUTION_OVERVIEW.md` for a problem-statement-to-implementation map and `DATA_DICTIONARY.md` for entity fields, relationships, and validation rules.

## Analysis rules and interpretation

- **Primary metric:** Conversion rate = assigned users with at least one conversion / assigned users.
- **Analysis population:** Intention-to-treat; conversions are attributed to original assignment regardless of exposure compliance.
- **Outcome window:** 30 days after assignment. Events beyond the window are flagged and excluded from the primary metric.
- **Treatment effect:** Absolute treatment-minus-control rate difference, relative uplift, and an approximate two-sided two-proportion z-test.
- **Uncertainty:** Wilson intervals for each arm and a normal-approximation 95% interval for the absolute difference.
- **SRM:** Chi-square goodness-of-fit against the experiment's configured A/B allocation.
- **Guardrails:** Recorded synthetic operational indicators, including errors and refunds, compared with predeclared thresholds.
- **Decision:** An evidence summary, not an automatic deployment action. A positive uplift is not sufficient if uncertainty, SRM, or guardrails indicate risk.

## Deliberate negative scenarios

The generator first creates the normal synthetic baseline, then injects five clearly labelled scenarios:

1. Multiple variants assigned to one user in an experiment.
2. Exposure recorded before assignment.
3. Conversion outside the outcome window.
4. Metric definition version effective after the result.
5. Sample-ratio mismatch.

Ground-truth labels are generated from these documented rules and stored separately from analytical input data.

## Important limitations

This is a synthetic hackathon prototype, not a production experimentation service.

Statistical estimates use transparent approximations and simulated guardrail signals. Production use would require:

- A reviewed analysis plan and stronger causal/statistical methods.
- Authentication and role-based access.
- Schema migrations.
- Append-only audit and event storage.
- Background processing and monitoring.
- Privacy and security review.

Regenerating the demo dataset replaces the current source-data snapshot; previously calculated experiment-result snapshots are retained and versioned in SQLite.

## Repository structure

```text
experimentiq/
├── app.py
├── requirements.txt
├── README.md
├── DATA_DICTIONARY.md
├── SOLUTION_OVERVIEW.md
├── src/
│   ├── config.py
│   ├── generator.py
│   ├── analytics.py
│   ├── validation.py
│   └── storage.py
├── tests/
│   └── test_core.py
└── data/
    └── experimentiq.db    # generated locally; database is git-ignored
```
