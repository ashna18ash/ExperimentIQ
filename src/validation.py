
"""Vectorized data-quality and relational-integrity checks."""

import numpy as np
import pandas as pd


# =============================================================
# CONFIGURATION
# =============================================================

VALID_EXPERIMENT_STATUSES = {
    "draft",
    "running",
    "completed",
    "paused",
}

VALID_TELEMETRY_STATUSES = {
    "captured",
    "missing",
}

VALID_GUARDRAIL_DIRECTIONS = {
    "max",
    "min",
}

SRM_P_VALUE_THRESHOLD = 0.001

PRIMARY_KEYS = {
    "experiments": "experiment_id",
    "variants": "variant_id",
    "audience_members": "user_id",
    "assignments": "assignment_id",
    "exposure_events": "exposure_id",
    "conversion_events": "conversion_id",
    "metric_definitions": "metric_definition_id",
    "guardrail_metrics": "guardrail_id",
    "guardrail_observations": "guardrail_observation_id",
    "experiment_results": "experiment_result_id",
}


# =============================================================
# SHARED HELPERS
# =============================================================

def _get_frame(frames, name):
    """Return a DataFrame even when a table is missing."""
    value = frames.get(name, pd.DataFrame())

    if isinstance(value, pd.DataFrame):
        return value

    return pd.DataFrame()


def _add_issue(issues, severity, code, entity, ids, detail):
    """Add validation issues while limiting excessive duplicate IDs."""

    if isinstance(ids, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        values = (
            pd.Series(ids)
            .dropna()
            .astype(str)
            .head(500)
            .unique()
        )

        for value in values:
            issues.append({
                "severity": severity,
                "issue_type": code,
                "entity": entity,
                "entity_id": value,
                "detail": detail,
            })

    else:
        issues.append({
            "severity": severity,
            "issue_type": code,
            "entity": entity,
            "entity_id": str(ids),
            "detail": detail,
        })


def _eligible_nonconflicting_assignments(assignments, audience=None):
    """
    Return assignments eligible for analytics and SRM.

    Excludes:
    1. Ineligible audience members.
    2. Users assigned to multiple distinct variants
       within the same experiment.

    All assignments belonging to a conflicting user are excluded.
    """

    if assignments.empty:
        return assignments.copy()

    required = {
        "experiment_id",
        "user_id",
        "variant_id",
    }

    if not required.issubset(assignments.columns):
        return assignments.iloc[0:0].copy()

    result = assignments.copy()

    if (
        audience is not None
        and not audience.empty
        and {"user_id", "eligible"}.issubset(audience.columns)
    ):
        eligibility = (
            audience[["user_id", "eligible"]]
            .drop_duplicates("user_id")
        )

        result = result.merge(
            eligibility,
            on="user_id",
            how="left",
            validate="many_to_one",
        )

        result = result.loc[
            result["eligible"].fillna(False).astype(bool)
        ].copy()

        result.drop(columns=["eligible"], inplace=True)

    conflict_counts = (
        result.groupby(
            ["experiment_id", "user_id"],
            dropna=False,
        )["variant_id"]
        .nunique()
    )

    conflict_keys = conflict_counts[
        conflict_counts > 1
    ].index

    if len(conflict_keys):
        assignment_keys = pd.MultiIndex.from_frame(
            result[["experiment_id", "user_id"]]
        )

        result = result.loc[
            ~assignment_keys.isin(conflict_keys)
        ].copy()

    return result


def _safe_datetime(series):
    """Convert timestamps to UTC without raising on invalid values."""
    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def _empty_issues_frame():
    """Return the stable validation issue schema."""
    return pd.DataFrame(
        columns=[
            "severity",
            "issue_type",
            "entity",
            "entity_id",
            "detail",
        ]
    )


# =============================================================
# MAIN DATA VALIDATION
# =============================================================

def validate_frames(frames):
    """
    Validate source DataFrames.

    Returns:
        DataFrame with columns:
        severity, issue_type, entity, entity_id, detail
    """

    issues = []

    def add(severity, code, entity, ids, detail):
        _add_issue(
            issues,
            severity,
            code,
            entity,
            ids,
            detail,
        )

    get = lambda name: _get_frame(frames, name)

    # ---------------------------------------------------------
    # 1. PRIMARY-KEY CHECKS
    # ---------------------------------------------------------

    for table, pk in PRIMARY_KEYS.items():
        df = get(table)

        if df.empty:
            continue

        if pk not in df.columns:
            add(
                "HIGH",
                "MISSING_PRIMARY_KEY_COLUMN",
                table,
                table,
                f"Required primary-key column {pk} is missing",
            )
            continue

        null_ids = df.index[df[pk].isna()]

        if len(null_ids):
            add(
                "HIGH",
                "NULL_PRIMARY_KEY",
                table,
                null_ids,
                f"Null values found in {pk}",
            )

        duplicates = df.loc[
            df[pk].duplicated(keep=False),
            pk,
        ]

        if not duplicates.empty:
            add(
                "HIGH",
                "DUPLICATE_PRIMARY_KEY",
                table,
                duplicates,
                f"Duplicate {pk}",
            )

    # ---------------------------------------------------------
    # 2. ASSIGNMENT FOREIGN KEYS AND VARIANT CONSISTENCY
    # ---------------------------------------------------------

    exp_ids = set(
        get("experiments")
        .get("experiment_id", pd.Series(dtype=str))
        .dropna()
        .astype(str)
    )

    user_ids = set(
        get("audience_members")
        .get("user_id", pd.Series(dtype=str))
        .dropna()
        .astype(str)
    )

    variant_ids = set(
        get("variants")
        .get("variant_id", pd.Series(dtype=str))
        .dropna()
        .astype(str)
    )

    assignments = get("assignments").copy()
    a = assignments

    if not a.empty:
        required_assignment_cols = [
            "assignment_id",
            "experiment_id",
            "user_id",
            "variant_id",
            "assigned_at",
        ]

        missing_cols = [
            col
            for col in required_assignment_cols
            if col not in a.columns
        ]

        if missing_cols:
            add(
                "HIGH",
                "MISSING_ASSIGNMENT_COLUMNS",
                "Assignment",
                "assignments",
                f"Missing columns: {missing_cols}",
            )

        else:
            for col, valid, code, detail in [
                (
                    "experiment_id",
                    exp_ids,
                    "ORPHAN_EXPERIMENT",
                    "Unknown experiment",
                ),
                (
                    "user_id",
                    user_ids,
                    "ORPHAN_USER",
                    "Unknown audience member",
                ),
                (
                    "variant_id",
                    variant_ids,
                    "ORPHAN_VARIANT",
                    "Unknown variant",
                ),
            ]:
                bad = a.loc[
                    ~a[col].astype(str).isin(valid),
                    "assignment_id",
                ]

                if not bad.empty:
                    add(
                        "HIGH",
                        code,
                        "Assignment",
                        bad,
                        detail,
                    )

            variants_df = get("variants")

            if {
                "variant_id",
                "experiment_id",
            }.issubset(variants_df.columns):

                assignment_variant = a.merge(
                    variants_df[
                        ["variant_id", "experiment_id"]
                    ],
                    on="variant_id",
                    how="left",
                    suffixes=("", "_variant"),
                )

                wrong_variant_exp = assignment_variant.loc[
                    assignment_variant["experiment_id"]
                    != assignment_variant["experiment_id_variant"],
                    "assignment_id",
                ]

                if not wrong_variant_exp.empty:
                    add(
                        "HIGH",
                        "VARIANT_EXPERIMENT_MISMATCH",
                        "Assignment",
                        wrong_variant_exp,
                        "Assigned variant belongs to a different experiment",
                    )

            # Detect users assigned to multiple variants.
            conflicts = (
                a.groupby(
                    ["experiment_id", "user_id"],
                    dropna=False,
                )["variant_id"]
                .nunique()
            )

            bad_keys = conflicts[conflicts > 1].index

            if len(bad_keys):
                assignment_keys = pd.MultiIndex.from_frame(
                    a[["experiment_id", "user_id"]]
                )

                conflict_mask = assignment_keys.isin(bad_keys)

                conflict_ids = a.loc[
                    conflict_mask,
                    "assignment_id",
                ]

                add(
                    "HIGH",
                    "MULTIPLE_VARIANT_ASSIGNMENT",
                    "Assignment",
                    conflict_ids,
                    "User assigned to multiple variants in one experiment",
                )

    # ---------------------------------------------------------
    # 3. EVENT PARENT CHECKS AND CHRONOLOGY
    # ---------------------------------------------------------

    if {
        "assignment_id",
        "assigned_at",
    }.issubset(a.columns):

        assignment_map = a[
            ["assignment_id", "assigned_at"]
        ].drop_duplicates("assignment_id")

    else:
        assignment_map = pd.DataFrame(
            columns=[
                "assignment_id",
                "assigned_at",
            ]
        )

    event_specs = [
        (
            "exposure_events",
            "exposure_id",
            "exposed_at",
            "EXPOSURE_BEFORE_ASSIGNMENT",
        ),
        (
            "conversion_events",
            "conversion_id",
            "converted_at",
            "OUTCOME_WINDOW_INCONSISTENT",
        ),
    ]

    for table, id_col, time_col, issue_code in event_specs:
        df = get(table).copy()

        if df.empty:
            continue

        required = [
            "assignment_id",
            id_col,
            time_col,
        ]

        if not set(required).issubset(df.columns):
            add(
                "HIGH",
                "MISSING_EVENT_COLUMNS",
                table,
                table,
                f"Required event columns missing: {required}",
            )
            continue

        merged = df.merge(
            assignment_map,
            on="assignment_id",
            how="left",
            indicator=True,
        )

        orphan = merged.loc[
            merged["_merge"] == "left_only",
            id_col,
        ]

        if not orphan.empty:
            add(
                "HIGH",
                "ORPHAN_ASSIGNMENT",
                table,
                orphan,
                "Assignment does not exist",
            )

        merged[time_col] = _safe_datetime(
            merged[time_col]
        )

        merged["assigned_at"] = _safe_datetime(
            merged["assigned_at"]
        )

        invalid_timestamp = merged.loc[
            (merged["_merge"] == "both")
            & (
                merged[time_col].isna()
                | merged["assigned_at"].isna()
            ),
            id_col,
        ]

        if not invalid_timestamp.empty:
            add(
                "HIGH",
                "INVALID_EVENT_TIMESTAMP",
                table,
                invalid_timestamp,
                "Event or assignment timestamp is invalid",
            )

        valid_parent = merged["_merge"] == "both"

        valid_times = (
            merged[time_col].notna()
            & merged["assigned_at"].notna()
        )

        if table == "exposure_events":
            bad = merged.loc[
                valid_parent
                & valid_times
                & (
                    merged[time_col]
                    < merged["assigned_at"]
                ),
                id_col,
            ]

            if not bad.empty:
                add(
                    "HIGH",
                    issue_code,
                    "ExposureEvent",
                    bad,
                    "Exposure timestamp precedes assignment",
                )

        else:
            bad = merged.loc[
                valid_parent
                & valid_times
                & (
                    (
                        merged[time_col]
                        < merged["assigned_at"]
                    )
                    | (
                        merged[time_col]
                        > merged["assigned_at"]
                        + pd.Timedelta(days=30)
                    )
                ),
                id_col,
            ]

            if not bad.empty:
                add(
                    "MEDIUM",
                    issue_code,
                    "ConversionEvent",
                    bad,
                    "Conversion is outside the valid 0-30 day outcome window",
                )

    # ---------------------------------------------------------
    # 4. METRIC DEFINITION CHANGES AFTER RESULT PUBLICATION
    # ---------------------------------------------------------

    metrics = get("metric_definitions")
    results = get("experiment_results")

    if (
        not metrics.empty
        and not results.empty
        and {
            "experiment_id",
            "generated_at",
        }.issubset(results.columns)
        and {
            "experiment_id",
            "effective_from",
            "version",
            "metric_definition_id",
        }.issubset(metrics.columns)
    ):
        result_dates = results.copy()

        result_dates["generated_at"] = _safe_datetime(
            result_dates["generated_at"]
        )

        first_results = (
            result_dates.groupby("experiment_id")["generated_at"]
            .min()
            .rename("first_result_at")
            .reset_index()
        )

        metric_dates = metrics.merge(
            first_results,
            on="experiment_id",
            how="inner",
        )

        metric_dates["effective_from_dt"] = _safe_datetime(
            metric_dates["effective_from"]
        )

        metric_dates["first_result_dt"] = _safe_datetime(
            metric_dates["first_result_at"]
        )

        versions = pd.to_numeric(
            metric_dates["version"],
            errors="coerce",
        )

        post = metric_dates.loc[
            versions.gt(1)
            & metric_dates["effective_from_dt"].notna()
            & metric_dates["first_result_dt"].notna()
            & (
                metric_dates["effective_from_dt"]
                > metric_dates["first_result_dt"]
            )
        ]

        if not post.empty:
            add(
                "MEDIUM",
                "METRIC_CHANGED_AFTER_RESULTS",
                "MetricDefinition",
                post["metric_definition_id"],
                "Metric definition became effective after the stored experiment result",
            )

    # ---------------------------------------------------------
    # 5. SAMPLE RATIO MISMATCH (SRM)
    # ---------------------------------------------------------

    experiments = get("experiments")
    variants = get("variants")
    audience = get("audience_members")

    if (
        not a.empty
        and not experiments.empty
        and not variants.empty
        and {
            "variant_id",
            "variant_key",
        }.issubset(variants.columns)
        and {
            "experiment_id",
            "allocation_a",
            "allocation_b",
        }.issubset(experiments.columns)
    ):
        srm_assignments = _eligible_nonconflicting_assignments(
            a,
            audience,
        )

        av = srm_assignments.merge(
            variants[
                ["variant_id", "variant_key"]
            ],
            on="variant_id",
            how="left",
        )

        counts = (
            av.groupby(
                ["experiment_id", "variant_key"]
            )
            .size()
            .unstack(fill_value=0)
        )

        from scipy.stats import chi2

        for _, experiment in experiments.iterrows():
            experiment_id = experiment["experiment_id"]

            if experiment_id not in counts.index:
                continue

            row = counts.loc[experiment_id]

            observed = np.array(
                [
                    row.get("A", 0),
                    row.get("B", 0),
                ],
                dtype=float,
            )

            total = float(observed.sum())

            if total <= 0:
                continue

            allocation = np.array(
                [
                    pd.to_numeric(
                        pd.Series([experiment["allocation_a"]]),
                        errors="coerce",
                    ).iloc[0],
                    pd.to_numeric(
                        pd.Series([experiment["allocation_b"]]),
                        errors="coerce",
                    ).iloc[0],
                ],
                dtype=float,
            )

            if (
                not np.isfinite(allocation).all()
                or (allocation <= 0).any()
                or not np.isclose(allocation.sum(), 1.0)
            ):
                add(
                    "HIGH",
                    "INVALID_EXPERIMENT_ALLOCATION",
                    "Experiment",
                    experiment_id,
                    "Configured allocation must be positive and sum to 1",
                )
                continue

            expected = allocation * total

            statistic = float(
                np.sum(
                    (observed - expected) ** 2
                    / np.maximum(expected, 1)
                )
            )

            p_value = float(
                chi2.sf(statistic, 1)
            )

            if p_value < SRM_P_VALUE_THRESHOLD:
                add(
                    "HIGH",
                    "SAMPLE_RATIO_MISMATCH",
                    "Experiment",
                    experiment_id,
                    (
                        "Observed allocation differs from configured "
                        f"{allocation[0]:.0%}/{allocation[1]:.0%} "
                        f"split (p={p_value:.3g})"
                    ),
                )

    # ---------------------------------------------------------
    # 6. CROSS-ENTITY CONSISTENCY FOR EVENTS
    # ---------------------------------------------------------

    for table, id_col in [
        ("exposure_events", "exposure_id"),
        ("conversion_events", "conversion_id"),
    ]:
        child = get(table)

        if child.empty or a.empty:
            continue

        parent_cols = [
            "assignment_id",
            "experiment_id",
            "user_id",
        ]

        if table == "exposure_events":
            parent_cols.append("variant_id")

        if not set(parent_cols).issubset(a.columns):
            continue

        if not {
            "assignment_id",
            id_col,
        }.issubset(child.columns):
            continue

        parent = a[parent_cols].drop_duplicates(
            "assignment_id"
        )

        joined = child.merge(
            parent,
            on="assignment_id",
            how="left",
            suffixes=("", "_parent"),
        )

        fields = (
            ["experiment_id", "user_id"]
            + (
                ["variant_id"]
                if table == "exposure_events"
                else []
            )
        )

        for field in fields:
            parent_field = f"{field}_parent"

            if (
                field not in joined.columns
                or parent_field not in joined.columns
            ):
                continue

            mismatch = joined.loc[
                joined[parent_field].notna()
                & joined[field].notna()
                & (
                    joined[field].astype(str)
                    != joined[parent_field].astype(str)
                ),
                id_col,
            ]

            if not mismatch.empty:
                add(
                    "HIGH",
                    "EVENT_PARENT_MISMATCH",
                    table,
                    mismatch,
                    f"{field} does not match referenced assignment",
                )

    # ---------------------------------------------------------
    # 7. EXPERIMENT LIFECYCLE AND DATE CHECKS
    # ---------------------------------------------------------

    exp_df = get("experiments")

    if not exp_df.empty:
        if {
            "status",
            "experiment_id",
        }.issubset(exp_df.columns):

            bad_status = exp_df.loc[
                ~exp_df["status"].isin(
                    VALID_EXPERIMENT_STATUSES
                ),
                "experiment_id",
            ]

            if not bad_status.empty:
                add(
                    "MEDIUM",
                    "INVALID_EXPERIMENT_STATUS",
                    "Experiment",
                    bad_status,
                    "Unknown lifecycle status",
                )

        if {
            "start_at",
            "end_at",
            "experiment_id",
        }.issubset(exp_df.columns):

            start = _safe_datetime(
                exp_df["start_at"]
            )

            end = _safe_datetime(
                exp_df["end_at"]
            )

            invalid_dates = exp_df.loc[
                start.isna() | end.isna(),
                "experiment_id",
            ]

            if not invalid_dates.empty:
                add(
                    "HIGH",
                    "INVALID_EXPERIMENT_TIMESTAMP",
                    "Experiment",
                    invalid_dates,
                    "Experiment start or end timestamp is missing or invalid",
                )

            bad_dates = exp_df.loc[
                start.notna()
                & end.notna()
                & (end < start),
                "experiment_id",
            ]

            if not bad_dates.empty:
                add(
                    "HIGH",
                    "INVALID_EXPERIMENT_DATES",
                    "Experiment",
                    bad_dates,
                    "Experiment end date precedes experiment start date",
                )

    # ---------------------------------------------------------
    # 8. AUDIENCE ELIGIBILITY
    # ---------------------------------------------------------

    if (
        not a.empty
        and not audience.empty
        and {
            "user_id",
            "eligible",
        }.issubset(audience.columns)
        and {
            "user_id",
            "assignment_id",
        }.issubset(a.columns)
    ):
        eligibility = (
            audience[
                ["user_id", "eligible"]
            ]
            .drop_duplicates("user_id")
        )

        exp_assignments = a.merge(
            eligibility,
            on="user_id",
            how="left",
            validate="many_to_one",
        )

        ineligible = exp_assignments.loc[
            ~exp_assignments["eligible"]
            .fillna(False)
            .astype(bool),
            "assignment_id",
        ]

        if not ineligible.empty:
            add(
                "HIGH",
                "INELIGIBLE_USER_ASSIGNED",
                "Assignment",
                ineligible,
                "Ineligible or unknown audience member received an assignment",
            )

    # ---------------------------------------------------------
    # 9. EXPOSURE TELEMETRY CHECKS
    # ---------------------------------------------------------

    exposure_df = get("exposure_events")

    if (
        not exposure_df.empty
        and "telemetry_status" in exposure_df.columns
    ):
        bad_telemetry = exposure_df.loc[
            ~exposure_df["telemetry_status"].isin(
                VALID_TELEMETRY_STATUSES
            ),
            "exposure_id",
        ] if "exposure_id" in exposure_df.columns else pd.Series(dtype=object)

        if not bad_telemetry.empty:
            add(
                "MEDIUM",
                "INVALID_TELEMETRY_STATUS",
                "ExposureEvent",
                bad_telemetry,
                "Unknown telemetry status",
            )

        if "exposure_id" in exposure_df.columns:
            missing = exposure_df.loc[
                exposure_df["telemetry_status"] == "missing",
                "exposure_id",
            ].head(100)

            if not missing.empty:
                add(
                    "LOW",
                    "MISSING_TELEMETRY",
                    "ExposureEvent",
                    missing,
                    "Exposure exists but telemetry payload is incomplete",
                )

    # ---------------------------------------------------------
    # 10. GUARDRAIL DEFINITION CHECKS
    # ---------------------------------------------------------

    guardrail_df = get("guardrail_metrics")

    if (
        not guardrail_df.empty
        and {
            "direction",
            "threshold",
            "guardrail_id",
        }.issubset(guardrail_df.columns)
    ):
        threshold = pd.to_numeric(
            guardrail_df["threshold"],
            errors="coerce",
        )

        bad_guard = guardrail_df.loc[
            ~guardrail_df["direction"].isin(
                VALID_GUARDRAIL_DIRECTIONS
            )
            | ~threshold.between(0, 1),
            "guardrail_id",
        ]

        if not bad_guard.empty:
            add(
                "HIGH",
                "INVALID_GUARDRAIL_DEFINITION",
                "GuardrailMetric",
                bad_guard,
                "Guardrail direction or proportion threshold is invalid",
            )

    # ---------------------------------------------------------
    # 11. GUARDRAIL OBSERVATION VALIDATION
    # ---------------------------------------------------------

    observations = get(
        "guardrail_observations"
    ).copy()

    if not observations.empty:
        required = [
            "guardrail_observation_id",
            "assignment_id",
            "experiment_id",
            "user_id",
            "error_occurred",
            "refund_occurred",
            "observed_at",
        ]

        missing_columns = [
            col
            for col in required
            if col not in observations.columns
        ]

        if missing_columns:
            add(
                "HIGH",
                "MISSING_GUARDRAIL_COLUMNS",
                "GuardrailObservation",
                "guardrail_observations",
                f"Missing required columns: {missing_columns}",
            )

        else:
            if {
                "assignment_id",
                "experiment_id",
                "user_id",
                "assigned_at",
            }.issubset(a.columns):

                parent = a[
                    [
                        "assignment_id",
                        "experiment_id",
                        "user_id",
                        "assigned_at",
                    ]
                ].drop_duplicates(
                    "assignment_id"
                )

                joined = observations.merge(
                    parent,
                    on="assignment_id",
                    how="left",
                    suffixes=("", "_parent"),
                    indicator=True,
                )

                orphan = joined.loc[
                    joined["_merge"] == "left_only",
                    "guardrail_observation_id",
                ]

                if not orphan.empty:
                    add(
                        "HIGH",
                        "ORPHAN_GUARDRAIL_ASSIGNMENT",
                        "GuardrailObservation",
                        orphan,
                        "Guardrail observation references an unknown assignment",
                    )

                for field in [
                    "experiment_id",
                    "user_id",
                ]:
                    parent_field = f"{field}_parent"

                    mismatch = joined.loc[
                        (joined["_merge"] == "both")
                        & joined[field].notna()
                        & joined[parent_field].notna()
                        & (
                            joined[field].astype(str)
                            != joined[parent_field].astype(str)
                        ),
                        "guardrail_observation_id",
                    ]

                    if not mismatch.empty:
                        add(
                            "HIGH",
                            "GUARDRAIL_PARENT_MISMATCH",
                            "GuardrailObservation",
                            mismatch,
                            f"{field} does not match referenced assignment",
                        )

                joined["observed_at_dt"] = _safe_datetime(
                    joined["observed_at"]
                )

                joined["assigned_at_dt"] = _safe_datetime(
                    joined["assigned_at"]
                )

                invalid_time = joined.loc[
                    (joined["_merge"] == "both")
                    & (
                        joined["observed_at_dt"].isna()
                        | joined["assigned_at_dt"].isna()
                        | (
                            joined["observed_at_dt"]
                            < joined["assigned_at_dt"]
                        )
                    ),
                    "guardrail_observation_id",
                ]

                if not invalid_time.empty:
                    add(
                        "HIGH",
                        "INVALID_GUARDRAIL_TIMESTAMP",
                        "GuardrailObservation",
                        invalid_time,
                        "Observation timestamp is invalid or precedes assignment",
                    )

            for field in [
                "error_occurred",
                "refund_occurred",
            ]:
                invalid = observations.loc[
                    ~observations[field].isin(
                        [True, False]
                    ),
                    "guardrail_observation_id",
                ]

                if not invalid.empty:
                    add(
                        "HIGH",
                        "INVALID_GUARDRAIL_VALUE",
                        "GuardrailObservation",
                        invalid,
                        f"{field} must contain Boolean values",
                    )

    # ---------------------------------------------------------
    # 12. GUARDRAIL OBSERVATION COVERAGE
    # ---------------------------------------------------------

    if (
        not a.empty
        and not observations.empty
        and {
            "assignment_id",
            "experiment_id",
            "user_id",
            "variant_id",
        }.issubset(a.columns)
        and "assignment_id" in observations.columns
    ):
        coverage_assignments = (
            _eligible_nonconflicting_assignments(
                a,
                audience,
            )
        )

        expected_ids = set(
            coverage_assignments["assignment_id"]
            .dropna()
            .astype(str)
        )

        observed_ids = set(
            observations["assignment_id"]
            .dropna()
            .astype(str)
        )

        missing_observation_ids = sorted(
            expected_ids - observed_ids
        )

        if missing_observation_ids:
            add(
                "MEDIUM",
                "MISSING_GUARDRAIL_OBSERVATION",
                "Assignment",
                missing_observation_ids,
                (
                    "Eligible assignment has no guardrail observation; "
                    "guardrail coverage is incomplete"
                ),
            )

    return pd.DataFrame(
        issues,
        columns=[
            "severity",
            "issue_type",
            "entity",
            "entity_id",
            "detail",
        ],
    ) if issues else _empty_issues_frame()


# =============================================================
# SEVERITY SUMMARY
# =============================================================

def severity_summary(issues):
    """
    Return validation issue counts by severity.

    Severity levels:
        CRITICAL, HIGH, MEDIUM, LOW
    """

    severity_order = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

    if (
        issues is None
        or issues.empty
        or "severity" not in issues.columns
    ):
        return pd.DataFrame({
            "severity": severity_order,
            "issue_count": [0, 0, 0, 0],
        })

    counts = (
        issues["severity"]
        .astype(str)
        .str.upper()
        .value_counts()
    )

    return pd.DataFrame({
        "severity": severity_order,
        "issue_count": [
            int(counts.get(level, 0))
            for level in severity_order
        ],
    })


# =============================================================
# DATABASE / RELATIONAL INTEGRITY SUMMARY
# =============================================================

def integrity_summary(frames):
    """
    Return primary-key, foreign-key, parent-consistency,
    and result-snapshot reconciliation checks.
    """

    checks = []

    # ---------------------------------------------------------
    # 1. PRIMARY-KEY CHECKS
    # ---------------------------------------------------------

    for table, key in PRIMARY_KEYS.items():
        df = _get_frame(frames, table)

        ok = (
            key in df.columns
            and not df[key].duplicated().any()
            and not df[key].isna().any()
        )

        checks.append({
            "check": (
                f"{table}: primary key uniqueness / non-null"
            ),
            "status": "PASS" if ok else "FAIL",
            "records": len(df),
        })

    # ---------------------------------------------------------
    # 2. FOREIGN-KEY CHECKS
    # ---------------------------------------------------------

    relations = [
        (
            "variants -> experiments",
            "variants",
            "experiment_id",
            "experiments",
            "experiment_id",
        ),
        (
            "assignments -> experiments",
            "assignments",
            "experiment_id",
            "experiments",
            "experiment_id",
        ),
        (
            "assignments -> audience",
            "assignments",
            "user_id",
            "audience_members",
            "user_id",
        ),
        (
            "assignments -> variants",
            "assignments",
            "variant_id",
            "variants",
            "variant_id",
        ),
        (
            "exposures -> assignments",
            "exposure_events",
            "assignment_id",
            "assignments",
            "assignment_id",
        ),
        (
            "conversions -> assignments",
            "conversion_events",
            "assignment_id",
            "assignments",
            "assignment_id",
        ),
        (
            "exposures -> variants",
            "exposure_events",
            "variant_id",
            "variants",
            "variant_id",
        ),
        (
            "metric definitions -> experiments",
            "metric_definitions",
            "experiment_id",
            "experiments",
            "experiment_id",
        ),
        (
            "guardrails -> experiments",
            "guardrail_metrics",
            "experiment_id",
            "experiments",
            "experiment_id",
        ),
        (
            "guardrail observations -> assignments",
            "guardrail_observations",
            "assignment_id",
            "assignments",
            "assignment_id",
        ),
        (
            "guardrail observations -> experiments",
            "guardrail_observations",
            "experiment_id",
            "experiments",
            "experiment_id",
        ),
        (
            "guardrail observations -> audience",
            "guardrail_observations",
            "user_id",
            "audience_members",
            "user_id",
        ),
        (
            "results -> experiments",
            "experiment_results",
            "experiment_id",
            "experiments",
            "experiment_id",
        ),
    ]

    for label, child, child_key, parent, parent_key in relations:
        child_df = _get_frame(frames, child)
        parent_df = _get_frame(frames, parent)

        ok = (
            child_key in child_df.columns
            and parent_key in parent_df.columns
            and child_df[child_key].notna().all()
            and child_df[child_key].astype(str).isin(
                parent_df[parent_key].dropna().astype(str)
            ).all()
        )

        checks.append({
            "check": f"{label}: foreign key integrity",
            "status": "PASS" if ok else "FAIL",
            "records": len(child_df),
        })

    # ---------------------------------------------------------
    # 3. GUARDRAIL OBSERVATION PARENT CONSISTENCY
    # ---------------------------------------------------------

    observations = _get_frame(
        frames,
        "guardrail_observations",
    )

    assignments = _get_frame(
        frames,
        "assignments",
    )

    if (
        not observations.empty
        and not assignments.empty
        and {
            "assignment_id",
            "experiment_id",
            "user_id",
        }.issubset(observations.columns)
        and {
            "assignment_id",
            "experiment_id",
            "user_id",
        }.issubset(assignments.columns)
    ):
        parent = assignments[
            [
                "assignment_id",
                "experiment_id",
                "user_id",
            ]
        ].drop_duplicates(
            "assignment_id"
        )

        joined = observations.merge(
            parent,
            on="assignment_id",
            how="left",
            suffixes=("", "_parent"),
        )

        consistent = (
            joined["experiment_id"].astype(str)
            == joined["experiment_id_parent"].astype(str)
        ) & (
            joined["user_id"].astype(str)
            == joined["user_id_parent"].astype(str)
        )

        checks.append({
            "check": "guardrail observations: parent consistency",
            "status": "PASS" if consistent.all() else "FAIL",
            "records": len(observations),
        })

    # ---------------------------------------------------------
    # 4. CURRENT RESULT SNAPSHOT RECONCILIATION
    # ---------------------------------------------------------

    results = _get_frame(
        frames,
        "experiment_results",
    )

    experiments = _get_frame(
        frames,
        "experiments",
    )

    audience = _get_frame(
        frames,
        "audience_members",
    )

    variants = _get_frame(
        frames,
        "variants",
    )

    if (
        not results.empty
        and {
            "dataset_seed",
            "dataset_size",
            "experiment_id",
            "control_assigned",
            "treatment_assigned",
        }.issubset(results.columns)
        and not experiments.empty
        and {
            "seed",
            "experiment_id",
        }.issubset(experiments.columns)
        and not audience.empty
        and "user_id" in audience.columns
    ):
        current_seed_values = pd.to_numeric(
            experiments["seed"],
            errors="coerce",
        ).dropna()

        if not current_seed_values.empty:
            current_seed = int(
                current_seed_values.iloc[0]
            )

            current_size = len(audience)

            result_seeds = pd.to_numeric(
                results["dataset_seed"],
                errors="coerce",
            )

            result_sizes = pd.to_numeric(
                results["dataset_size"],
                errors="coerce",
            )

            current_results = results.loc[
                (result_seeds == current_seed)
                & (result_sizes == current_size)
            ].copy()

            a = _get_frame(
                frames,
                "assignments",
            ).copy()

            required_a = {
                "experiment_id",
                "user_id",
                "variant_id",
            }

            required_v = {
                "variant_id",
                "variant_key",
            }

            if (
                not current_results.empty
                and not a.empty
                and required_a.issubset(a.columns)
                and required_v.issubset(variants.columns)
            ):
                # Use the same eligibility and conflict filters
                # as analytics and SRM.
                a = _eligible_nonconflicting_assignments(
                    a,
                    audience,
                )

                av = a.merge(
                    variants[
                        [
                            "variant_id",
                            "variant_key",
                        ]
                    ],
                    on="variant_id",
                    how="left",
                )

                counts = (
                    av.groupby(
                        [
                            "experiment_id",
                            "variant_key",
                        ]
                    )
                    .size()
                )

                reconciliation_ok = True

                for _, result in current_results.iterrows():
                    experiment_id = result["experiment_id"]

                    actual_a = int(
                        counts.get(
                            (experiment_id, "A"),
                            0,
                        )
                    )

                    actual_b = int(
                        counts.get(
                            (experiment_id, "B"),
                            0,
                        )
                    )

                    expected_a = int(
                        result["control_assigned"]
                    )

                    expected_b = int(
                        result["treatment_assigned"]
                    )

                    if (
                        actual_a != expected_a
                        or actual_b != expected_b
                    ):
                        reconciliation_ok = False
                        break

                checks.append({
                    "check": (
                        "current result snapshots: "
                        "assigned-count reconciliation"
                    ),
                    "status": (
                        "PASS"
                        if reconciliation_ok
                        else "FAIL"
                    ),
                    "records": len(current_results),
                })

    return pd.DataFrame(
        checks,
        columns=[
            "check",
            "status",
            "records",
        ],
    )
