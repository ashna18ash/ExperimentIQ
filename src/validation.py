
"""Vectorized data-quality and relational-integrity checks."""

import numpy as np
import pandas as pd


def validate_frames(frames):
    issues = []

    def add(severity, code, entity, ids, detail):
        if isinstance(ids, (pd.Series, pd.Index, np.ndarray, list, tuple)):
            values = pd.Series(ids).dropna().astype(str).head(500).unique()
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

    get = lambda name: frames.get(name, pd.DataFrame())

    # 1. Primary-key checks
    pk_map = {
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

    for table, pk in pk_map.items():
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

        null_ids = df.loc[df[pk].isna()].index
        if len(null_ids):
            add(
                "HIGH",
                "NULL_PRIMARY_KEY",
                table,
                null_ids,
                f"Null values found in {pk}",
            )

        dup = df.loc[df[pk].duplicated(keep=False), pk]
        if not dup.empty:
            add(
                "HIGH",
                "DUPLICATE_PRIMARY_KEY",
                table,
                dup,
                f"Duplicate {pk}",
            )

    # 2. Assignment foreign keys and variant consistency
    exp_ids = set(
        get("experiments").get(
            "experiment_id", pd.Series(dtype=str)
        ).dropna().astype(str)
    )

    user_ids = set(
        get("audience_members").get(
            "user_id", pd.Series(dtype=str)
        ).dropna().astype(str)
    )

    variant_ids = set(
        get("variants").get(
            "variant_id", pd.Series(dtype=str)
        ).dropna().astype(str)
    )

    a = get("assignments").copy()

    if not a.empty:
        required_assignment_cols = [
            "assignment_id",
            "experiment_id",
            "user_id",
            "variant_id",
            "assigned_at",
        ]

        missing_cols = [
            col for col in required_assignment_cols
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
                    add("HIGH", code, "Assignment", bad, detail)

            variants_df = get("variants")

            if {
                "variant_id", "experiment_id"
            }.issubset(variants_df.columns):
                assignment_variant = a.merge(
                    variants_df[["variant_id", "experiment_id"]],
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

            # A user must not be assigned to multiple variants
            # within the same experiment.
            conflicts = (
                a.groupby(
                    ["experiment_id", "user_id"],
                    dropna=False,
                )["variant_id"]
                .nunique()
            )

            bad_keys = conflicts[conflicts > 1].index

            if len(bad_keys):
                conflict_ids = (
                    a.set_index(["experiment_id", "user_id"])
                    .loc[bad_keys, "assignment_id"]
                )

                add(
                    "HIGH",
                    "MULTIPLE_VARIANT_ASSIGNMENT",
                    "Assignment",
                    conflict_ids,
                    "User assigned to multiple variants in one experiment",
                )

    # 3. Event parent checks and chronology
    if {"assignment_id", "assigned_at"}.issubset(a.columns):
        amap = a[
            ["assignment_id", "assigned_at"]
        ].drop_duplicates("assignment_id")
    else:
        amap = pd.DataFrame(
            columns=["assignment_id", "assigned_at"]
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

        required = ["assignment_id", id_col, time_col]

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
            amap,
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

        merged[time_col] = pd.to_datetime(
            merged[time_col],
            errors="coerce",
            utc=True,
        )

        merged["assigned_at"] = pd.to_datetime(
            merged["assigned_at"],
            errors="coerce",
            utc=True,
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

        if table == "exposure_events":
            bad = merged.loc[
                (merged["_merge"] == "both")
                & merged[time_col].notna()
                & merged["assigned_at"].notna()
                & (
                    merged[time_col] < merged["assigned_at"]
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
                (merged["_merge"] == "both")
                & merged[time_col].notna()
                & merged["assigned_at"].notna()
                & (
                    (merged[time_col] < merged["assigned_at"])
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

    # 4. Metric definition changes after result publication
    metrics = get("metric_definitions")
    results = get("experiment_results")

    if (
        not metrics.empty
        and not results.empty
        and {
            "experiment_id", "generated_at"
        }.issubset(results.columns)
        and {
            "experiment_id", "effective_from",
            "version", "metric_definition_id",
        }.issubset(metrics.columns)
    ):
        res = (
            results.groupby("experiment_id")["generated_at"]
            .min()
            .rename("first_result_at")
            .reset_index()
        )

        mm = metrics.merge(
            res,
            on="experiment_id",
            how="inner",
        )

        mm["effective_from_dt"] = pd.to_datetime(
            mm["effective_from"],
            errors="coerce",
            utc=True,
        )

        mm["first_result_dt"] = pd.to_datetime(
            mm["first_result_at"],
            errors="coerce",
            utc=True,
        )

        post = mm.loc[
            (mm["version"] > 1)
            & (mm["effective_from_dt"] > mm["first_result_dt"])
        ]

        if not post.empty:
            add(
                "MEDIUM",
                "METRIC_CHANGED_AFTER_RESULTS",
                "MetricDefinition",
                post["metric_definition_id"],
                "Metric definition became effective after the stored experiment result",
            )

    # 5. Sample Ratio Mismatch (SRM)
    experiments = get("experiments")
    variants = get("variants")

    if (
        not a.empty
        and not experiments.empty
        and not variants.empty
        and {
            "variant_id", "variant_key"
        }.issubset(variants.columns)
        and {
            "experiment_id", "allocation_a", "allocation_b"
        }.issubset(experiments.columns)
    ):
        av = a.merge(
            variants[["variant_id", "variant_key"]],
            on="variant_id",
            how="left",
        )

        counts = (
            av.groupby(["experiment_id", "variant_key"])
            .size()
            .unstack(fill_value=0)
        )

        from scipy.stats import chi2

        for _, er in experiments.iterrows():
            if er["experiment_id"] not in counts.index:
                continue

            row = counts.loc[er["experiment_id"]]

            total = float(
                row.get("A", 0) + row.get("B", 0)
            )

            if total <= 0:
                continue

            obs = np.array(
                [row.get("A", 0), row.get("B", 0)],
                dtype=float,
            )

            allocation = np.array(
                [
                    float(er["allocation_a"]),
                    float(er["allocation_b"]),
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
                    er["experiment_id"],
                    "Configured allocation must be positive and sum to 1",
                )
                continue

            expected = allocation * total

            stat = float(
                np.sum(
                    (obs - expected) ** 2
                    / np.maximum(expected, 1)
                )
            )

            p = float(chi2.sf(stat, 1))

            if p < 0.001:
                add(
                    "HIGH",
                    "SAMPLE_RATIO_MISMATCH",
                    "Experiment",
                    er["experiment_id"],
                    (
                        "Observed allocation differs from configured "
                        f"{allocation[0]:.0%}/{allocation[1]:.0%} "
                        f"split (p={p:.3g})"
                    ),
                )

    # 6. Cross-entity consistency for exposure/conversion events
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

        parent = a[parent_cols].drop_duplicates(
            "assignment_id"
        )

        joined = child.merge(
            parent,
            on="assignment_id",
            how="left",
            suffixes=("", "_parent"),
        )

        for field in (
            ["experiment_id", "user_id"]
            + (
                ["variant_id"]
                if table == "exposure_events"
                else []
            )
        ):
            parent_field = f"{field}_parent"

            if field not in joined.columns:
                continue

            mismatch = joined.loc[
                joined[parent_field].notna()
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

    # 7. Experiment lifecycle and date checks
    exp_df = get("experiments")

    if not exp_df.empty:
        if {"status", "experiment_id"}.issubset(exp_df.columns):
            bad_status = exp_df.loc[
                ~exp_df["status"].isin(
                    ["draft", "running", "completed", "paused"]
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

        if {"start_at", "end_at", "experiment_id"}.issubset(exp_df.columns):
            start = pd.to_datetime(
                exp_df["start_at"],
                errors="coerce",
                utc=True,
            )

            end = pd.to_datetime(
                exp_df["end_at"],
                errors="coerce",
                utc=True,
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

    # 8. Audience eligibility
    audience = get("audience_members")

    if (
        not a.empty
        and not audience.empty
        and {"user_id", "eligible"}.issubset(audience.columns)
    ):
        exp_assign = a.merge(
            audience[["user_id", "eligible"]],
            on="user_id",
            how="left",
        )

        ineligible = exp_assign.loc[
            exp_assign["eligible"].fillna(False) == False,
            "assignment_id",
        ]

        if not ineligible.empty:
            add(
                "HIGH",
                "INELIGIBLE_USER_ASSIGNED",
                "Assignment",
                ineligible,
                "Ineligible audience member received an assignment",
            )

    # 9. Exposure telemetry checks
    exposure_df = get("exposure_events")

    if (
        not exposure_df.empty
        and "telemetry_status" in exposure_df.columns
    ):
        bad_telemetry = exposure_df.loc[
            ~exposure_df["telemetry_status"].isin(
                ["captured", "missing"]
            ),
            "exposure_id",
        ]

        if not bad_telemetry.empty:
            add(
                "MEDIUM",
                "INVALID_TELEMETRY_STATUS",
                "ExposureEvent",
                bad_telemetry,
                "Unknown telemetry status",
            )

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

    # 10. Guardrail definition checks
    guardrail_df = get("guardrail_metrics")

    if (
        not guardrail_df.empty
        and {
            "direction", "threshold", "guardrail_id"
        }.issubset(guardrail_df.columns)
    ):
        threshold = pd.to_numeric(
            guardrail_df["threshold"],
            errors="coerce",
        )

        bad_guard = guardrail_df.loc[
            ~guardrail_df["direction"].isin(["max", "min"])
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

    # 11. Guardrail observation validation
    observations = get("guardrail_observations").copy()

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
            col for col in required
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
                "assignment_id", "experiment_id",
                "user_id", "assigned_at",
            }.issubset(a.columns):
                parent = a[
                    [
                        "assignment_id",
                        "experiment_id",
                        "user_id",
                        "assigned_at",
                    ]
                ].drop_duplicates("assignment_id")

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

                for field in ["experiment_id", "user_id"]:
                    parent_field = f"{field}_parent"

                    mismatch = joined.loc[
                        (joined["_merge"] == "both")
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

                joined["observed_at_dt"] = pd.to_datetime(
                    joined["observed_at"],
                    errors="coerce",
                    utc=True,
                )

                joined["assigned_at_dt"] = pd.to_datetime(
                    joined["assigned_at"],
                    errors="coerce",
                    utc=True,
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

            for field in ["error_occurred", "refund_occurred"]:
                invalid = observations.loc[
                    ~observations[field].isin([True, False]),
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

    return pd.DataFrame(
        issues,
        columns=[
            "severity",
            "issue_type",
            "entity",
            "entity_id",
            "detail",
        ],
    )


# =============================================================
# DATABASE / RELATIONAL INTEGRITY SUMMARY
# =============================================================

def integrity_summary(frames):
    checks = []

    # 1. Primary-key checks
    keys = {
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

    for table, key in keys.items():
        df = frames.get(table, pd.DataFrame())

        ok = (
            key in df.columns
            and not df[key].duplicated().any()
            and not df[key].isna().any()
        )

        checks.append({
            "check": f"{table}: primary key uniqueness / non-null",
            "status": "PASS" if ok else "FAIL",
            "records": len(df),
        })

    # 2. Foreign-key checks
    relations = [
        ("variants -> experiments", "variants", "experiment_id",
         "experiments", "experiment_id"),
        ("assignments -> experiments", "assignments", "experiment_id",
         "experiments", "experiment_id"),
        ("assignments -> audience", "assignments", "user_id",
         "audience_members", "user_id"),
        ("assignments -> variants", "assignments", "variant_id",
         "variants", "variant_id"),
        ("exposures -> assignments", "exposure_events", "assignment_id",
         "assignments", "assignment_id"),
        ("conversions -> assignments", "conversion_events", "assignment_id",
         "assignments", "assignment_id"),
        ("exposures -> variants", "exposure_events", "variant_id",
         "variants", "variant_id"),
        ("metric definitions -> experiments", "metric_definitions",
         "experiment_id", "experiments", "experiment_id"),
        ("guardrails -> experiments", "guardrail_metrics",
         "experiment_id", "experiments", "experiment_id"),
        ("guardrail observations -> assignments", "guardrail_observations",
         "assignment_id", "assignments", "assignment_id"),
        ("guardrail observations -> experiments", "guardrail_observations",
         "experiment_id", "experiments", "experiment_id"),
        ("guardrail observations -> audience", "guardrail_observations",
         "user_id", "audience_members", "user_id"),
        ("results -> experiments", "experiment_results",
         "experiment_id", "experiments", "experiment_id"),
    ]

    for label, child, child_key, parent, parent_key in relations:
        child_df = frames.get(child, pd.DataFrame())
        parent_df = frames.get(parent, pd.DataFrame())

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

    # 3. Guardrail observation parent consistency
    observations = frames.get(
        "guardrail_observations",
        pd.DataFrame(),
    )

    assignments = frames.get(
        "assignments",
        pd.DataFrame(),
    )

    if (
        not observations.empty
        and not assignments.empty
        and {
            "assignment_id", "experiment_id", "user_id"
        }.issubset(observations.columns)
        and {
            "assignment_id", "experiment_id", "user_id"
        }.issubset(assignments.columns)
    ):
        parent = assignments[
            ["assignment_id", "experiment_id", "user_id"]
        ].drop_duplicates("assignment_id")

        joined = observations.merge(
            parent,
            on="assignment_id",
            how="left",
            suffixes=("", "_parent"),
            validate="many_to_one",
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

    # 4. Reconcile source assignments with current result snapshots.
    # Historical snapshots from earlier dataset sizes are retained.
    results = frames.get(
        "experiment_results",
        pd.DataFrame(),
    )

    experiments = frames.get(
        "experiments",
        pd.DataFrame(),
    )

    audience = frames.get(
        "audience_members",
        pd.DataFrame(),
    )

    variants = frames.get(
        "variants",
        pd.DataFrame(),
    )

    if (
        not results.empty
        and {
            "dataset_seed", "dataset_size",
            "experiment_id", "control_assigned",
            "treatment_assigned",
        }.issubset(results.columns)
        and not experiments.empty
        and {"seed", "experiment_id"}.issubset(experiments.columns)
        and not audience.empty
        and "user_id" in audience.columns
    ):
        current_seed = int(experiments["seed"].iloc[0])
        current_size = len(audience)

        current_results = results.loc[
            (pd.to_numeric(results["dataset_seed"], errors="coerce")
             == current_seed)
            & (pd.to_numeric(results["dataset_size"], errors="coerce")
               == current_size)
        ].copy()

        a = frames.get("assignments", pd.DataFrame()).copy()

        required_a = {
            "experiment_id", "user_id", "variant_id"
        }

        required_v = {"variant_id", "variant_key"}

        if (
            not current_results.empty
            and not a.empty
            and required_a.issubset(a.columns)
            and required_v.issubset(variants.columns)
        ):
            # Match analytics eligibility filtering.
            if "is_eligible" in a.columns:
                a = a.loc[
                    a["is_eligible"].fillna(False).astype(bool)
                ].copy()

            elif "eligible" in a.columns:
                a = a.loc[
                    a["eligible"].fillna(False).astype(bool)
                ].copy()

            elif {"user_id", "eligible"}.issubset(audience.columns):
                eligibility = audience[
                    ["user_id", "eligible"]
                ].drop_duplicates("user_id")

                a = a.merge(
                    eligibility,
                    on="user_id",
                    how="left",
                    validate="many_to_one",
                )

                a = a.loc[
                    a["eligible"].fillna(False).astype(bool)
                ].copy()

            # Match analytics conflict handling:
            # exclude every assignment belonging to a user who has
            # multiple distinct variants in the same experiment.
            conflict_counts = (
                a.groupby(
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
                    a[["experiment_id", "user_id"]]
                )

                conflicting_mask = assignment_keys.isin(
                    conflict_keys
                )

                a = a.loc[~conflicting_mask].copy()

            # Add variant labels for control/treatment counts.
            av = a.merge(
                variants[["variant_id", "variant_key"]],
                on="variant_id",
                how="left",
                validate="many_to_one",
            )

            counts = (
                av.groupby(
                    ["experiment_id", "variant_key"]
                )
                .size()
            )

            reconciliation_ok = True

            for _, result in current_results.iterrows():
                experiment_id = result["experiment_id"]

                actual_a = int(
                    counts.get((experiment_id, "A"), 0)
                )

                actual_b = int(
                    counts.get((experiment_id, "B"), 0)
                )

                expected_a = int(result["control_assigned"])
                expected_b = int(result["treatment_assigned"])

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
                "status": "PASS" if reconciliation_ok else "FAIL",
                "records": len(current_results),
            })

    return pd.DataFrame(checks)
