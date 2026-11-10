"""Statistical analysis for ExperimentIQ synthetic experimentation data."""

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2


ANALYSIS_WINDOW_DAYS = 30
DEFAULT_ANALYSIS_AT = "2026-10-31T23:59:00+00:00"


def wilson_interval(successes, total, alpha=0.05):
    """Calculate a Wilson confidence interval for a proportion."""
    if total <= 0:
        return (0.0, 0.0)

    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(
        (p * (1 - p) / total) + (z * z / (4 * total * total))
    ) / den

    return (
        max(0.0, center - half),
        min(1.0, center + half),
    )


def _empty_guardrail_frame():
    return pd.DataFrame(columns=[
        "metric",
        "control_observed",
        "treatment_observed",
        "threshold",
        "direction",
        "unit",
        "treatment_minus_control",
        "status",
    ])


def _select_metric_definition(metric_definitions, exp_id, analysis_at):
    """Select the latest metric definition effective at the analysis time."""
    if metric_definitions is None or metric_definitions.empty:
        return None

    definitions = metric_definitions.copy()

    if "experiment_id" in definitions.columns:
        definitions = definitions.loc[
            definitions["experiment_id"] == exp_id
        ].copy()

    if definitions.empty:
        return None

    effective_column = next(
        (
            column for column in
            ["effective_from", "effective_at", "valid_from"]
            if column in definitions.columns
        ),
        None,
    )

    if effective_column:
        definitions["_effective_dt"] = pd.to_datetime(
            definitions[effective_column],
            errors="coerce",
            utc=True,
        )

        analysis_dt = pd.to_datetime(analysis_at, utc=True)

        definitions = definitions.loc[
            definitions["_effective_dt"].notna()
            & (definitions["_effective_dt"] <= analysis_dt)
        ].copy()

        if definitions.empty:
            return None

        definitions = definitions.sort_values("_effective_dt")

    return definitions.iloc[-1].to_dict()


def _calculate_guardrails(
    exp_id,
    assignments,
    guardrail_definitions,
    guardrail_observations,
):
    """Calculate actual control and treatment guardrail rates."""
    if (
        guardrail_observations is None
        or guardrail_observations.empty
        or assignments.empty
        or guardrail_definitions is None
        or guardrail_definitions.empty
    ):
        return _empty_guardrail_frame()

    observations = guardrail_observations.copy()
    definitions = guardrail_definitions.loc[
        guardrail_definitions["experiment_id"] == exp_id
    ].copy()

    if definitions.empty:
        return _empty_guardrail_frame()

    # Only use observations belonging to assignments in this experiment.
    assignment_columns = [
        column for column in ["assignment_id", "variant_key"]
        if column in assignments.columns
    ]

    assignment_lookup = assignments[assignment_columns].drop_duplicates(
        "assignment_id"
    )

    observations = observations.merge(
        assignment_lookup,
        on="assignment_id",
        how="inner",
        validate="many_to_one",
    )

    rows = []

    indicator_columns = {
        "error_rate": "error_occurred",
        "refund_rate": "refund_occurred",
    }

    for _, definition in definitions.iterrows():
        metric_name = str(definition["metric_name"])
        indicator = indicator_columns.get(metric_name)

        if indicator is None or indicator not in observations.columns:
            continue

        metric_data = observations.loc[
            observations["variant_key"].isin(["A", "B"])
        ].copy()

        metric_data[indicator] = (
            metric_data[indicator]
            .fillna(False)
            .astype(bool)
        )

        control = metric_data.loc[
            metric_data["variant_key"] == "A", indicator
        ]
        treatment = metric_data.loc[
            metric_data["variant_key"] == "B", indicator
        ]

        control_rate = float(control.mean()) if len(control) else 0.0
        treatment_rate = float(treatment.mean()) if len(treatment) else 0.0

        threshold = float(definition["threshold"])
        direction = str(definition["direction"]).lower()

        if not len(control) or not len(treatment):
            status = "NOT AVAILABLE"
        elif direction == "max":
            status = (
                "FAIL" if treatment_rate > threshold else "PASS"
            )
        elif direction == "min":
            status = (
                "FAIL" if treatment_rate < threshold else "PASS"
            )
        else:
            status = "INVALID DEFINITION"

        rows.append({
            "metric": metric_name,
            "control_observed": control_rate,
            "treatment_observed": treatment_rate,
            "threshold": threshold,
            "direction": direction,
            "unit": definition.get("unit", "rate"),
            "treatment_minus_control": treatment_rate - control_rate,
            "status": status,
        })

    return pd.DataFrame(rows) if rows else _empty_guardrail_frame()


def analyze_experiment(
    exp_id,
    assignments,
    conversions,
    exposures,
    variants,
    guardrails,
    experiments=None,
    guardrail_observations=None,
    metric_definitions=None,
    analysis_at=DEFAULT_ANALYSIS_AT,
):
    """
    Analyze one experiment.

    Existing positional arguments are retained for backward compatibility.
    Additional data sources are optional.
    """

    # ---------------------------------------------------------
    # 1. Select experiment records
    # ---------------------------------------------------------
    a = assignments.loc[
        assignments["experiment_id"] == exp_id
    ].copy()

    c = conversions.loc[
        conversions["experiment_id"] == exp_id
    ].copy()

    v = variants.loc[
        variants["experiment_id"] == exp_id,
        ["variant_id", "variant_key", "name"],
    ].copy()

    a = a.merge(
        v,
        on="variant_id",
        how="left",
        validate="many_to_one",
    )

    original_assignment_count = len(a)

    # ---------------------------------------------------------
    # 2. Exclude ineligible assignments, when eligibility exists
    # ---------------------------------------------------------
    eligibility_column = next(
        (
            column for column in ["is_eligible", "eligible"]
            if column in a.columns
        ),
        None,
    )

    if eligibility_column:
        eligible_mask = (
            a[eligibility_column]
            .fillna(False)
            .astype(bool)
        )
        a = a.loc[eligible_mask].copy()

    # ---------------------------------------------------------
    # 3. Identify conflicting assignments
    # ---------------------------------------------------------
    conflicting_users = set()

    if not a.empty and {"user_id", "variant_id"}.issubset(a.columns):
        variants_per_user = (
            a.groupby("user_id")["variant_id"]
            .nunique()
        )

        conflicting_users = set(
            variants_per_user[variants_per_user > 1].index
        )

    if conflicting_users:
        a = a.loc[
            ~a["user_id"].isin(conflicting_users)
        ].copy()

    excluded_conflict_assignments = (
        original_assignment_count
        - len(a)
    )

    # ---------------------------------------------------------
    # 4. Validate and attribute conversion events
    # ---------------------------------------------------------
    valid_conversions = pd.DataFrame(
        columns=["assignment_id"]
    )

    if not a.empty and not c.empty:
        assignment_lookup_columns = [
            "assignment_id",
            "assigned_at",
            "experiment_id",
        ]

        if "user_id" in a.columns:
            assignment_lookup_columns.append("user_id")

        assignment_lookup = a[
            assignment_lookup_columns
        ].drop_duplicates("assignment_id")

        valid_conversions = c.merge(
            assignment_lookup,
            on="assignment_id",
            how="inner",
            suffixes=("_conversion", "_assignment"),
            validate="many_to_one",
        )

        # Reject conversion records whose user or experiment does not
        # match the parent assignment.
        if (
            "user_id_conversion" in valid_conversions.columns
            and "user_id_assignment" in valid_conversions.columns
        ):
            valid_conversions = valid_conversions.loc[
                valid_conversions["user_id_conversion"]
                == valid_conversions["user_id_assignment"]
            ]

        if (
            "experiment_id_conversion" in valid_conversions.columns
            and "experiment_id_assignment" in valid_conversions.columns
        ):
            valid_conversions = valid_conversions.loc[
                valid_conversions["experiment_id_conversion"]
                == valid_conversions["experiment_id_assignment"]
            ]

        valid_conversions["converted_at_dt"] = pd.to_datetime(
            valid_conversions["converted_at"],
            errors="coerce",
            utc=True,
        )

        valid_conversions["assigned_at_dt"] = pd.to_datetime(
            valid_conversions["assigned_at"],
            errors="coerce",
            utc=True,
        )

        valid_conversions = valid_conversions.loc[
            valid_conversions["converted_at_dt"].notna()
            & valid_conversions["assigned_at_dt"].notna()
            & (
                valid_conversions["converted_at_dt"]
                >= valid_conversions["assigned_at_dt"]
            )
            & (
                valid_conversions["converted_at_dt"]
                <= valid_conversions["assigned_at_dt"]
                + pd.Timedelta(days=ANALYSIS_WINDOW_DAYS)
            )
        ].copy()

        # Count each assignment as converted at most once.
        valid_conversions = valid_conversions.drop_duplicates(
            subset=["assignment_id"]
        )

    converted_assignment_ids = set(
        valid_conversions["assignment_id"].astype(str)
    )

    a["converted"] = (
        a["assignment_id"].astype(str)
        .isin(converted_assignment_ids)
    )

    # ---------------------------------------------------------
    # 5. Calculate control and treatment conversion rates
    # ---------------------------------------------------------
    arms = []

    for arm in ["A", "B"]:
        x = a.loc[a["variant_key"] == arm]

        n = len(x)
        k = int(x["converted"].sum()) if n else 0

        lower, upper = wilson_interval(k, n)

        arms.append({
            "variant": arm,
            "name": "Control" if arm == "A" else "Treatment",
            "assigned": n,
            "conversions": k,
            "conversion_rate": k / n if n else 0.0,
            "ci_low": lower,
            "ci_high": upper,
        })

    out = pd.DataFrame(arms)

    # ---------------------------------------------------------
    # 6. Calculate uplift, significance and confidence interval
    # ---------------------------------------------------------
    if (
        len(out) == 2
        and out.loc[0, "assigned"] > 0
        and out.loc[1, "assigned"] > 0
    ):
        p_a = float(out.loc[0, "conversion_rate"])
        p_b = float(out.loc[1, "conversion_rate"])

        n_a = int(out.loc[0, "assigned"])
        n_b = int(out.loc[1, "assigned"])

        k_a = int(out.loc[0, "conversions"])
        k_b = int(out.loc[1, "conversions"])

        pooled = (k_a + k_b) / (n_a + n_b)

        standard_error = np.sqrt(
            max(
                pooled * (1 - pooled) * (1 / n_a + 1 / n_b),
                1e-15,
            )
        )

        z_score = (p_b - p_a) / standard_error
        p_value = float(2 * norm.sf(abs(z_score)))

        uplift = (
            (p_b - p_a) / p_a * 100
            if p_a > 0
            else 0.0
        )

        difference_se = np.sqrt(
            max(
                p_a * (1 - p_a) / n_a
                + p_b * (1 - p_b) / n_b,
                0,
            )
        )

        difference_ci = (
            p_b - p_a - 1.96 * difference_se,
            p_b - p_a + 1.96 * difference_se,
        )

    else:
        uplift = 0.0
        p_value = 1.0
        difference_ci = (0.0, 0.0)

    # ---------------------------------------------------------
    # 7. Sample Ratio Mismatch against declared allocation
    # ---------------------------------------------------------
    expected_a = 0.5
    expected_b = 0.5

    if experiments is not None:
        experiment_rows = experiments.loc[
            experiments["experiment_id"] == exp_id
        ]

        if not experiment_rows.empty:
            expected_a = float(
                experiment_rows.iloc[0]["allocation_a"]
            )
            expected_b = float(
                experiment_rows.iloc[0]["allocation_b"]
            )

    observed = np.array([
        int((a["variant_key"] == "A").sum()),
        int((a["variant_key"] == "B").sum()),
    ], dtype=float)

    total = float(observed.sum())

    expected = np.array([
        expected_a,
        expected_b,
    ]) * total

    srm_chi_square = (
        float(
            np.sum(
                (observed - expected) ** 2
                / np.maximum(expected, 1)
            )
        )
        if total > 0
        else 0.0
    )

    srm_p_value = (
        float(chi2.sf(srm_chi_square, 1))
        if total > 0
        else 1.0
    )

    srm_status = (
        "FAIL" if srm_p_value < 0.001 else "PASS"
    )

    # ---------------------------------------------------------
    # 8. Calculate guardrails from generated observation records
    # ---------------------------------------------------------
    guardrail_df = _calculate_guardrails(
        exp_id=exp_id,
        assignments=a,
        guardrail_definitions=guardrails,
        guardrail_observations=guardrail_observations,
    )

    assessed_guardrails = guardrail_df.loc[
        guardrail_df["status"].isin(["PASS", "FAIL"])
    ] if not guardrail_df.empty else guardrail_df

    guardrails_pass = (
        assessed_guardrails.empty
        or (assessed_guardrails["status"] == "PASS").all()
    )

    # ---------------------------------------------------------
    # 9. Select the metric definition effective at analysis time
    # ---------------------------------------------------------
    selected_metric = _select_metric_definition(
        metric_definitions,
        exp_id,
        analysis_at,
    )

    metric_definition_id = None
    metric_version = None
    metric_name = "conversion_rate"

    if selected_metric:
        metric_definition_id = selected_metric.get(
            "metric_definition_id"
        )
        metric_version = selected_metric.get("version")
        metric_name = selected_metric.get(
            "metric_name",
            "conversion_rate",
        )

    # ---------------------------------------------------------
    # 10. Evidence-based decision
    # ---------------------------------------------------------
    significant = p_value < 0.05

    if srm_status == "FAIL":
        decision = "DO NOT ROLL OUT - SAMPLE RATIO MISMATCH"

    elif not guardrails_pass:
        decision = "DO NOT ROLL OUT - GUARDRAIL FAILURE"

    elif significant and uplift > 0:
        decision = "PROMISING - REVIEW"

    elif significant and uplift < 0:
        decision = "TREATMENT UNDERPERFORMS"

    else:
        decision = "INCONCLUSIVE"

    # ---------------------------------------------------------
    # 11. Return analysis results
    # ---------------------------------------------------------
    return {
        "variants": out,
        "uplift_pct": float(uplift),
        "p_value": float(p_value),
        "ci_low": float(difference_ci[0]),
        "ci_high": float(difference_ci[1]),
        "srm_p_value": srm_p_value,
        "srm_status": srm_status,
        "expected_allocation_a": expected_a,
        "expected_allocation_b": expected_b,
        "guardrails": guardrail_df,
        "assigned_total": int(total),
        "decision": decision,
        "excluded_conflict_assignments": int(
            excluded_conflict_assignments
        ),
        "conflicting_users": int(len(conflicting_users)),
        "valid_conversion_events": int(
            len(valid_conversions)
        ),
        "analysis_window_days": ANALYSIS_WINDOW_DAYS,
        "metric_name": metric_name,
        "metric_definition_id": metric_definition_id,
        "metric_version": metric_version,
    }


def all_experiment_summaries(frames):
    """Create one analysis summary per experiment."""
    rows = []

    exp_names = frames["experiments"].set_index(
        "experiment_id"
    )["name"]

    for exp_id in frames["experiments"]["experiment_id"]:
        result = analyze_experiment(
            exp_id,
            frames["assignments"],
            frames["conversion_events"],
            frames["exposure_events"],
            frames["variants"],
            frames["guardrail_metrics"],
            frames["experiments"],
            guardrail_observations=frames.get(
                "guardrail_observations"
            ),
            metric_definitions=frames.get(
                "metric_definitions"
            ),
        )

        rows.append({
            "experiment_id": exp_id,
            "name": exp_names.loc[exp_id],
            "assigned": result["assigned_total"],
            "uplift_pct": result["uplift_pct"],
            "p_value": result["p_value"],
            "srm_p_value": result["srm_p_value"],
            "srm_status": result["srm_status"],
            "guardrails_status": (
                "FAIL"
                if (
                    not result["guardrails"].empty
                    and (
                        result["guardrails"]["status"] == "FAIL"
                    ).any()
                )
                else "PASS"
            ),
            "excluded_conflict_assignments": (
                result["excluded_conflict_assignments"]
            ),
            "decision": result["decision"],
        })

    return pd.DataFrame(rows)


def build_result_records(
    frames,
    generated_at=DEFAULT_ANALYSIS_AT,
):
    """Persist one immutable result snapshot per experiment."""
    rows = []

    for exp_id in frames["experiments"]["experiment_id"]:
        result = analyze_experiment(
            exp_id,
            frames["assignments"],
            frames["conversion_events"],
            frames["exposure_events"],
            frames["variants"],
            frames["guardrail_metrics"],
            frames["experiments"],
            guardrail_observations=frames.get(
                "guardrail_observations"
            ),
            metric_definitions=frames.get(
                "metric_definitions"
            ),
            analysis_at=generated_at,
        )

        arms = result["variants"].set_index("variant")

        rows.append({
            "experiment_result_id": f"RES-{exp_id}-V1",
            "experiment_id": exp_id,
            "analysis_plan_version": "1.0",
            "generated_at": generated_at,
            "dataset_seed": int(
                frames["experiments"]["seed"].iloc[0]
            ),
            "dataset_size": int(
                len(frames["audience_members"])
            ),
            "control_assigned": int(
                arms.loc["A", "assigned"]
            ),
            "treatment_assigned": int(
                arms.loc["B", "assigned"]
            ),
            "control_conversions": int(
                arms.loc["A", "conversions"]
            ),
            "treatment_conversions": int(
                arms.loc["B", "conversions"]
            ),
            "control_rate": float(
                arms.loc["A", "conversion_rate"]
            ),
            "treatment_rate": float(
                arms.loc["B", "conversion_rate"]
            ),
            "uplift_pct": result["uplift_pct"],
            "p_value": result["p_value"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "srm_status": result["srm_status"],
            "decision": result["decision"],
        })

    return pd.DataFrame(rows)