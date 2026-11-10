
"""Statistical analysis for ExperimentIQ synthetic experimentation data."""

import numpy as np
import pandas as pd

from scipy.optimize import brentq
from scipy.stats import chi2, norm


ANALYSIS_WINDOW_DAYS = 30
DEFAULT_ANALYSIS_AT = "2026-10-31T23:59:00+00:00"

DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80
DEFAULT_ASSUMED_RELATIVE_EFFECT = 0.10
GUARDRAIL_MINIMUM_COVERAGE = 1.0


# =============================================================
# 1. Statistical helper functions
# =============================================================

def wilson_interval(successes, total, alpha=DEFAULT_ALPHA):
    """Calculate a Wilson confidence interval for a proportion."""
    if total <= 0:
        return (np.nan, np.nan)

    if not 0 <= successes <= total:
        return (np.nan, np.nan)

    z = norm.ppf(1 - alpha / 2)
    p = successes / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator

    half_width = (
        z
        * np.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def newcombe_difference_interval(
    successes_a=None,
    total_a=None,
    successes_b=None,
    total_b=None,
    alpha=DEFAULT_ALPHA,
    **kwargs,
):
    """
    Newcombe confidence interval for treatment minus control.

    Uses Wilson intervals for the two independent proportions.

    Supports both parameter conventions:

    A/B:
        successes_a, total_a, successes_b, total_b

    Control/treatment:
        successes_control, total_control,
        successes_treatment, total_treatment

    Positional calls remain compatible.
    """

    # Backwards compatibility with control/treatment keyword names.
    successes_a = kwargs.pop(
        "successes_control",
        successes_a,
    )

    total_a = kwargs.pop(
        "total_control",
        total_a,
    )

    successes_b = kwargs.pop(
        "successes_treatment",
        successes_b,
    )

    total_b = kwargs.pop(
        "total_treatment",
        total_b,
    )

    if kwargs:
        unexpected = ", ".join(kwargs.keys())
        raise TypeError(
            f"Unexpected arguments: {unexpected}"
        )

    if (
        successes_a is None
        or total_a is None
        or successes_b is None
        or total_b is None
    ):
        raise TypeError(
            "Both groups require successes and total values."
        )

    if (
        total_a <= 0
        or total_b <= 0
        or not 0 <= successes_a <= total_a
        or not 0 <= successes_b <= total_b
    ):
        return (np.nan, np.nan)

    p_a = successes_a / total_a
    p_b = successes_b / total_b

    a_low, a_high = wilson_interval(
        successes_a,
        total_a,
        alpha,
    )

    b_low, b_high = wilson_interval(
        successes_b,
        total_b,
        alpha,
    )

    difference = p_b - p_a

    lower_distance = np.sqrt(
        (p_b - b_low) ** 2
        + (a_high - p_a) ** 2
    )

    upper_distance = np.sqrt(
        (b_high - p_b) ** 2
        + (p_a - a_low) ** 2
    )

    return (
        max(-1.0, difference - lower_distance),
        min(1.0, difference + upper_distance),
    )


def _normal_approximation_power(
    p_control,
    p_treatment,
    n_control,
    n_treatment,
    alpha=DEFAULT_ALPHA,
):
    """Calculate approximate two-sided power for two proportions."""
    if n_control <= 0 or n_treatment <= 0:
        return np.nan

    if (
        not 0 < p_control < 1
        or not 0 < p_treatment < 1
    ):
        return np.nan

    pooled = (
        n_control * p_control
        + n_treatment * p_treatment
    ) / (n_control + n_treatment)

    null_se = np.sqrt(
        pooled * (1 - pooled)
        * (1 / n_control + 1 / n_treatment)
    )

    alternative_se = np.sqrt(
        p_control * (1 - p_control) / n_control
        + p_treatment * (1 - p_treatment) / n_treatment
    )

    if null_se <= 0 or alternative_se <= 0:
        return np.nan

    critical_value = norm.ppf(1 - alpha / 2)
    difference = abs(p_treatment - p_control)

    power = (
        norm.sf(
            critical_value * null_se / alternative_se
            - difference / alternative_se
        )
        + norm.cdf(
            -critical_value * null_se / alternative_se
            - difference / alternative_se
        )
    )

    return float(np.clip(power, 0.0, 1.0))


def _required_sample_size_per_arm(
    p_control,
    relative_effect=DEFAULT_ASSUMED_RELATIVE_EFFECT,
    target_power=DEFAULT_TARGET_POWER,
    alpha=DEFAULT_ALPHA,
):
    """Estimate required equal sample size per arm."""
    if not 0 < p_control < 1:
        return None

    if relative_effect <= 0 or not 0 < target_power < 1:
        return None

    p_treatment = min(
        p_control * (1 + relative_effect),
        0.999999,
    )

    difference = p_treatment - p_control

    if difference <= 0:
        return None

    pooled = (p_control + p_treatment) / 2

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(target_power)

    numerator = (
        z_alpha * np.sqrt(2 * pooled * (1 - pooled))
        + z_power
        * np.sqrt(
            p_control * (1 - p_control)
            + p_treatment * (1 - p_treatment)
        )
    ) ** 2

    return int(np.ceil(numerator / difference**2))


def _calculate_power_metrics(
    p_control,
    n_control,
    n_treatment,
    alpha=DEFAULT_ALPHA,
    target_power=DEFAULT_TARGET_POWER,
    assumed_relative_effect=DEFAULT_ASSUMED_RELATIVE_EFFECT,
):
    """
    Calculate MDE, estimated power and required sample size.

    MDE is the minimum detectable positive absolute change at the
    requested power, using a normal approximation.
    """
    result = {
        "baseline_conversion_rate": (
            float(p_control)
            if np.isfinite(p_control)
            else None
        ),
        "assumed_relative_effect_pct": (
            float(assumed_relative_effect * 100)
        ),
        "target_power": float(target_power),
        "alpha": float(alpha),
        "statistical_power": None,
        "mde_absolute": None,
        "mde_pct": None,
        "required_sample_size_per_arm": None,
        "sample_size_sufficient": None,
    }

    if (
        not np.isfinite(p_control)
        or n_control <= 0
        or n_treatment <= 0
        or not 0 < p_control < 1
    ):
        return result

    assumed_treatment_rate = min(
        p_control * (1 + assumed_relative_effect),
        0.999999,
    )

    result["statistical_power"] = _normal_approximation_power(
        p_control,
        assumed_treatment_rate,
        n_control,
        n_treatment,
        alpha,
    )

    result["required_sample_size_per_arm"] = (
        _required_sample_size_per_arm(
            p_control,
            assumed_relative_effect,
            target_power,
            alpha,
        )
    )

    if result["required_sample_size_per_arm"] is not None:
        result["sample_size_sufficient"] = bool(
            n_control >= result["required_sample_size_per_arm"]
            and n_treatment >= result["required_sample_size_per_arm"]
        )

    # Find the smallest positive absolute effect detectable at the
    # requested power for the current sample sizes.
    maximum_effect = 0.999999 - p_control

    if maximum_effect > 0:

        def power_difference(effect):
            return (
                _normal_approximation_power(
                    p_control,
                    p_control + effect,
                    n_control,
                    n_treatment,
                    alpha,
                )
                - target_power
            )

        try:
            if power_difference(maximum_effect) >= 0:
                mde = brentq(
                    power_difference,
                    1e-10,
                    maximum_effect,
                )

                result["mde_absolute"] = float(mde)
                result["mde_pct"] = float(
                    mde / p_control * 100
                )

        except (
            ValueError,
            RuntimeError,
            ZeroDivisionError,
        ):
            pass

    return result


# =============================================================
# 2. Metric definition versioning
# =============================================================

def _select_metric_definition(
    metric_definitions,
    exp_id,
    analysis_at,
):
    """Select the metric definition active at the analysis time."""
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
            column
            for column in [
                "effective_from",
                "effective_at",
                "valid_from",
            ]
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

        analysis_dt = pd.to_datetime(
            analysis_at,
            errors="coerce",
            utc=True,
        )

        if pd.isna(analysis_dt):
            return None

        definitions = definitions.loc[
            definitions["_effective_dt"].notna()
            & (definitions["_effective_dt"] <= analysis_dt)
        ].copy()

        if "effective_to" in definitions.columns:
            effective_to = pd.to_datetime(
                definitions["effective_to"],
                errors="coerce",
                utc=True,
            )

            definitions = definitions.loc[
                effective_to.isna()
                | (effective_to >= analysis_dt)
            ].copy()

        if definitions.empty:
            return None

        definitions = definitions.sort_values(
            "_effective_dt"
        )

    return definitions.iloc[-1].to_dict()


def _validate_itt_metric_definition(definition):
    """
    Check whether the selected definition matches assigned-user ITT.

    Exposed-user or other unsupported denominators must not silently
    receive an assigned-user conversion-rate calculation.
    """
    if definition is None:
        return False, "METRIC DEFINITION UNAVAILABLE"

    numerator = str(
        definition.get("numerator", "")
    ).strip().lower().replace(" ", "_")

    denominator = str(
        definition.get("denominator", "")
    ).strip().lower().replace(" ", "_")

    metric_name = str(
        definition.get("metric_name", "")
    ).strip().lower()

    # When numerator/denominator are not provided, accept only the
    # existing generic conversion_rate definition.
    if not numerator and not denominator:
        if metric_name in {
            "conversion_rate",
            "primary_conversion_rate",
        }:
            return True, "SUPPORTED ITT DEFINITION"

        return False, "UNSUPPORTED METRIC DEFINITION"

    numerator_supported = (
        "convert" in numerator
        and (
            "assigned" in numerator
            or numerator in {
                "converted_users",
                "conversions",
                "converted_users_30d",
            }
        )
    )

    denominator_supported = (
        "assigned" in denominator
        or denominator in {
            "assigned_users",
            "eligible_assigned_users",
        }
    )

    if numerator_supported and denominator_supported:
        return True, "SUPPORTED ITT DEFINITION"

    return False, "UNSUPPORTED METRIC DEFINITION"


# =============================================================
# 3. Guardrail analysis
# =============================================================

def _empty_guardrail_frame():
    return pd.DataFrame(columns=[
        "metric",
        "control_observed",
        "treatment_observed",
        "threshold",
        "direction",
        "unit",
        "treatment_minus_control",
        "control_expected",
        "treatment_expected",
        "control_observed_count",
        "treatment_observed_count",
        "control_coverage",
        "treatment_coverage",
        "overall_coverage",
        "status",
    ])


def _calculate_guardrails(
    exp_id,
    assignments,
    guardrail_definitions,
    guardrail_observations,
):
    """
    Calculate guardrail rates and observation coverage.

    Missing telemetry is not treated as a zero-valued observation.
    A guardrail is assessed only when both arms have complete coverage.
    """
    if (
        assignments is None
        or assignments.empty
        or guardrail_definitions is None
        or guardrail_definitions.empty
    ):
        return _empty_guardrail_frame()

    definitions = guardrail_definitions.loc[
        guardrail_definitions["experiment_id"] == exp_id
    ].copy()

    if definitions.empty:
        return _empty_guardrail_frame()

    if (
        guardrail_observations is None
        or guardrail_observations.empty
    ):
        observations = pd.DataFrame(
            columns=["assignment_id"]
        )
    else:
        observations = guardrail_observations.copy()

    assignment_lookup = assignments[
        ["assignment_id", "variant_key"]
    ].drop_duplicates("assignment_id")

    if not observations.empty:
        observations = observations.merge(
            assignment_lookup,
            on="assignment_id",
            how="inner",
            validate="many_to_one",
        )

    indicator_columns = {
        "error_rate": "error_occurred",
        "refund_rate": "refund_occurred",
    }

    rows = []

    for _, definition in definitions.iterrows():
        metric_name = str(
            definition.get("metric_name", "")
        )

        indicator = indicator_columns.get(metric_name)

        threshold = pd.to_numeric(
            pd.Series([definition.get("threshold")]),
            errors="coerce",
        ).iloc[0]

        direction = str(
            definition.get("direction", "")
        ).lower()

        control_expected = int(
            (assignments["variant_key"] == "A").sum()
        )

        treatment_expected = int(
            (assignments["variant_key"] == "B").sum()
        )

        if (
            indicator is None
            or indicator not in observations.columns
        ):
            metric_data = pd.DataFrame()
        else:
            metric_data = observations[
                observations["variant_key"].isin(["A", "B"])
            ].copy()

            # Duplicate observations cannot inflate coverage.
            metric_data = metric_data.drop_duplicates(
                subset=["assignment_id"],
                keep="last",
            )

        if (
            metric_data.empty
            or indicator not in metric_data.columns
        ):
            control_values = pd.Series(dtype=float)
            treatment_values = pd.Series(dtype=float)
        else:
            metric_data[indicator] = pd.to_numeric(
                metric_data[indicator],
                errors="coerce",
            )

            control_values = metric_data.loc[
                metric_data["variant_key"] == "A",
                indicator,
            ].dropna()

            treatment_values = metric_data.loc[
                metric_data["variant_key"] == "B",
                indicator,
            ].dropna()

        control_observed_count = len(control_values)
        treatment_observed_count = len(treatment_values)

        control_coverage = (
            control_observed_count / control_expected
            if control_expected
            else 0.0
        )

        treatment_coverage = (
            treatment_observed_count / treatment_expected
            if treatment_expected
            else 0.0
        )

        total_expected = (
            control_expected + treatment_expected
        )

        total_observed = (
            control_observed_count + treatment_observed_count
        )

        overall_coverage = (
            total_observed / total_expected
            if total_expected
            else 0.0
        )

        control_rate = (
            float(control_values.mean())
            if control_observed_count
            else np.nan
        )

        treatment_rate = (
            float(treatment_values.mean())
            if treatment_observed_count
            else np.nan
        )

        if (
            control_expected == 0
            or treatment_expected == 0
            or control_observed_count == 0
            or treatment_observed_count == 0
        ):
            status = "NOT AVAILABLE"

        elif (
            control_coverage < GUARDRAIL_MINIMUM_COVERAGE
            or treatment_coverage < GUARDRAIL_MINIMUM_COVERAGE
        ):
            status = "INSUFFICIENT DATA"

        elif (
            pd.isna(threshold)
            or direction not in {"min", "max"}
        ):
            status = "INVALID DEFINITION"

        elif direction == "max":
            status = (
                "FAIL"
                if treatment_rate > float(threshold)
                else "PASS"
            )

        else:
            status = (
                "FAIL"
                if treatment_rate < float(threshold)
                else "PASS"
            )

        rows.append({
            "metric": metric_name,
            "control_observed": control_rate,
            "treatment_observed": treatment_rate,
            "threshold": (
                float(threshold)
                if pd.notna(threshold)
                else np.nan
            ),
            "direction": direction,
            "unit": definition.get("unit", "rate"),
            "treatment_minus_control": (
                treatment_rate - control_rate
                if (
                    np.isfinite(control_rate)
                    and np.isfinite(treatment_rate)
                )
                else np.nan
            ),
            "control_expected": control_expected,
            "treatment_expected": treatment_expected,
            "control_observed_count": control_observed_count,
            "treatment_observed_count": treatment_observed_count,
            "control_coverage": float(control_coverage),
            "treatment_coverage": float(treatment_coverage),
            "overall_coverage": float(overall_coverage),
            "status": status,
        })

    return (
        pd.DataFrame(rows)
        if rows
        else _empty_guardrail_frame()
    )


def _overall_guardrail_status(guardrail_df):
    """Return an honest overall guardrail evidence status."""
    if guardrail_df is None or guardrail_df.empty:
        return "NOT AVAILABLE"

    statuses = set(
        guardrail_df["status"].astype(str)
    )

    if "FAIL" in statuses:
        return "FAIL"

    if "INVALID DEFINITION" in statuses:
        return "INVALID DEFINITION"

    if "INSUFFICIENT DATA" in statuses:
        return "INSUFFICIENT DATA"

    if "NOT AVAILABLE" in statuses:
        return "NOT AVAILABLE"

    if statuses == {"PASS"}:
        return "PASS"

    return "NOT AVAILABLE"


# =============================================================
# 4. Conversion event quality
# =============================================================

def _analyze_conversion_events(
    exp_id,
    all_assignments,
    conversions,
    eligible_assignments,
):
    """
    Attribute conversions and report event-quality problems separately.

    Counts are event-record counts, except valid_conversions, which is
    the number of distinct converted assignments.
    """
    counts = {
        "valid_conversion_events": 0,
        "duplicate_conversion_events": 0,
        "invalid_conversion_events": 0,
        "mismatched_conversion_events": 0,
        "orphan_conversion_events": 0,
    }

    valid = pd.DataFrame(
        columns=["assignment_id"]
    )

    if conversions is None or conversions.empty:
        return valid, counts

    if all_assignments is None or all_assignments.empty:
        candidates = conversions.loc[
            conversions["experiment_id"] == exp_id
        ].copy()

        counts["orphan_conversion_events"] = len(
            candidates
        )

        return valid, counts

    experiment_assignment_ids = set(
        all_assignments.loc[
            all_assignments["experiment_id"] == exp_id,
            "assignment_id",
        ].astype(str)
    )

    # Include records tagged for this experiment and records linked to
    # one of its assignments, even if their experiment_id is incorrect.
    candidate_mask = (
        conversions["experiment_id"].eq(exp_id)
        | conversions["assignment_id"].astype(str).isin(
            experiment_assignment_ids
        )
    )

    candidates = conversions.loc[
        candidate_mask
    ].copy()

    if candidates.empty:
        return valid, counts

    parent_columns = [
        "assignment_id",
        "experiment_id",
        "user_id",
        "assigned_at",
    ]

    parent = all_assignments[
        [
            column
            for column in parent_columns
            if column in all_assignments.columns
        ]
    ].drop_duplicates("assignment_id")

    merged = candidates.merge(
        parent,
        on="assignment_id",
        how="left",
        suffixes=("_conversion", "_assignment"),
        indicator=True,
        validate="many_to_one",
    )

    orphan_mask = merged["_merge"] == "left_only"

    counts["orphan_conversion_events"] = int(
        orphan_mask.sum()
    )

    matched = merged.loc[
        ~orphan_mask
    ].copy()

    mismatch_mask = pd.Series(
        False,
        index=matched.index,
    )

    if (
        "experiment_id_conversion" in matched.columns
        and "experiment_id_assignment" in matched.columns
    ):
        mismatch_mask |= (
            matched["experiment_id_conversion"]
            != matched["experiment_id_assignment"]
        )

    if (
        "user_id_conversion" in matched.columns
        and "user_id_assignment" in matched.columns
    ):
        mismatch_mask |= (
            matched["user_id_conversion"]
            != matched["user_id_assignment"]
        )

    counts["mismatched_conversion_events"] = int(
        mismatch_mask.sum()
    )

    matched = matched.loc[
        ~mismatch_mask
    ].copy()

    if matched.empty:
        return valid, counts

    # Count repeated records per assignment before deduplication.
    counts["duplicate_conversion_events"] = int(
        matched.duplicated(
            subset=["assignment_id"],
            keep="first",
        ).sum()
    )

    matched["converted_at_dt"] = pd.to_datetime(
        matched["converted_at"],
        errors="coerce",
        utc=True,
    )

    matched["assigned_at_dt"] = pd.to_datetime(
        matched["assigned_at"],
        errors="coerce",
        utc=True,
    )

    invalid_mask = (
        matched["converted_at_dt"].isna()
        | matched["assigned_at_dt"].isna()
        | (
            matched["converted_at_dt"]
            < matched["assigned_at_dt"]
        )
        | (
            matched["converted_at_dt"]
            > matched["assigned_at_dt"]
            + pd.Timedelta(days=ANALYSIS_WINDOW_DAYS)
        )
    )

    counts["invalid_conversion_events"] = int(
        invalid_mask.sum()
    )

    matched = matched.loc[
        ~invalid_mask
    ].copy()

    # Only assignments retained in the primary ITT population can
    # contribute to the conversion-rate numerator.
    retained_ids = set(
        eligible_assignments["assignment_id"].astype(str)
    )

    matched = matched.loc[
        matched["assignment_id"].astype(str).isin(
            retained_ids
        )
    ].copy()

    valid = matched.drop_duplicates(
        subset=["assignment_id"],
        keep="first",
    )

    counts["valid_conversion_events"] = int(
        valid["assignment_id"].nunique()
    )

    return valid, counts


# =============================================================
# 5. Main experiment analysis
# =============================================================

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
    alpha=DEFAULT_ALPHA,
    target_power=DEFAULT_TARGET_POWER,
    assumed_relative_effect=DEFAULT_ASSUMED_RELATIVE_EFFECT,
):
    """
    Analyze one experiment using an intention-to-treat approach.

    Existing positional arguments are retained for compatibility.

    Exposures are deliberately not required for ITT conversion analysis.
    Exposure telemetry is evaluated separately in activity and quality
    reporting.
    """

    # ---------------------------------------------------------
    # 1. Select experiment records and attach variant information
    # ---------------------------------------------------------
    raw_assignments = assignments.loc[
        assignments["experiment_id"] == exp_id
    ].copy()

    v = variants.loc[
        variants["experiment_id"] == exp_id,
        ["variant_id", "variant_key", "name"],
    ].copy()

    a = raw_assignments.merge(
        v,
        on="variant_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # 2. Separate ineligible assignments
    # ---------------------------------------------------------
    eligibility_column = next(
        (
            column
            for column in ["is_eligible", "eligible"]
            if column in a.columns
        ),
        None,
    )

    ineligible_assignment_count = 0

    if eligibility_column:
        eligible_mask = (
            a[eligibility_column]
            .fillna(False)
            .astype(bool)
        )

        ineligible_assignment_count = int(
            (~eligible_mask).sum()
        )

        a = a.loc[
            eligible_mask
        ].copy()

    # ---------------------------------------------------------
    # 3. Exclude users assigned to multiple variants
    # ---------------------------------------------------------
    conflicting_users = set()

    if (
        not a.empty
        and {"user_id", "variant_id"}.issubset(a.columns)
    ):
        variants_per_user = (
            a.groupby("user_id")["variant_id"].nunique()
        )

        conflicting_users = set(
            variants_per_user[
                variants_per_user > 1
            ].index
        )

    conflicting_assignment_count = 0

    if conflicting_users:
        conflict_mask = a["user_id"].isin(
            conflicting_users
        )

        conflicting_assignment_count = int(
            conflict_mask.sum()
        )

        a = a.loc[
            ~conflict_mask
        ].copy()

    # Preserve the legacy key while correcting its meaning.
    excluded_conflict_assignments = (
        conflicting_assignment_count
    )

    # ---------------------------------------------------------
    # 4. Validate and attribute conversion events
    # ---------------------------------------------------------
    valid_conversions, conversion_quality = (
        _analyze_conversion_events(
            exp_id=exp_id,
            all_assignments=assignments,
            conversions=conversions,
            eligible_assignments=a,
        )
    )

    converted_assignment_ids = set(
        valid_conversions["assignment_id"].astype(str)
    )

    a["converted"] = (
        a["assignment_id"].astype(str).isin(
            converted_assignment_ids
        )
    )

    # ---------------------------------------------------------
    # 5. Calculate arm-level conversion rates and Wilson intervals
    # ---------------------------------------------------------
    arms = []

    for arm in ["A", "B"]:
        arm_data = a.loc[
            a["variant_key"] == arm
        ]

        n = len(arm_data)

        k = (
            int(arm_data["converted"].sum())
            if n
            else 0
        )

        lower, upper = wilson_interval(
            k,
            n,
            alpha,
        )

        arms.append({
            "variant": arm,
            "name": (
                "Control"
                if arm == "A"
                else "Treatment"
            ),
            "assigned": n,
            "conversions": k,
            "conversion_rate": (
                k / n
                if n
                else 0.0
            ),
            "ci_low": lower,
            "ci_high": upper,
        })

    out = pd.DataFrame(arms)

    # ---------------------------------------------------------
    # 6. Uplift, significance and Newcombe confidence interval
    # ---------------------------------------------------------
    n_a = int(out.loc[0, "assigned"])
    n_b = int(out.loc[1, "assigned"])

    k_a = int(out.loc[0, "conversions"])
    k_b = int(out.loc[1, "conversions"])

    p_a = float(out.loc[0, "conversion_rate"])
    p_b = float(out.loc[1, "conversion_rate"])

    if n_a > 0 and n_b > 0:
        pooled = (
            k_a + k_b
        ) / (
            n_a + n_b
        )

        standard_error = np.sqrt(
            max(
                pooled
                * (1 - pooled)
                * (1 / n_a + 1 / n_b),
                1e-15,
            )
        )

        z_score = (
            p_b - p_a
        ) / standard_error

        p_value = float(
            2 * norm.sf(abs(z_score))
        )

        uplift = (
            (p_b - p_a) / p_a * 100
            if p_a > 0
            else np.nan
        )

        difference_ci = newcombe_difference_interval(
            k_a,
            n_a,
            k_b,
            n_b,
            alpha,
        )

    else:
        p_value = np.nan
        uplift = np.nan
        difference_ci = (
            np.nan,
            np.nan,
        )

    # ---------------------------------------------------------
    # 7. MDE, power and required sample size
    # ---------------------------------------------------------
    power_metrics = _calculate_power_metrics(
        p_control=(
            p_a
            if n_a > 0
            else np.nan
        ),
        n_control=n_a,
        n_treatment=n_b,
        alpha=alpha,
        target_power=target_power,
        assumed_relative_effect=assumed_relative_effect,
    )

    # ---------------------------------------------------------
    # 8. Sample Ratio Mismatch on the retained ITT population
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

    allocation_valid = (
        np.isfinite(expected_a)
        and np.isfinite(expected_b)
        and expected_a > 0
        and expected_b > 0
        and np.isclose(
            expected_a + expected_b,
            1.0,
        )
    )

    observed = np.array(
        [n_a, n_b],
        dtype=float,
    )

    total = float(
        observed.sum()
    )

    if allocation_valid and total > 0:
        expected = np.array([
            expected_a,
            expected_b,
        ]) * total

        srm_chi_square = float(
            np.sum(
                (observed - expected) ** 2
                / np.maximum(expected, 1e-12)
            )
        )

        srm_p_value = float(
            chi2.sf(srm_chi_square, 1)
        )

        srm_status = (
            "FAIL"
            if srm_p_value < 0.001
            else "PASS"
        )

    else:
        srm_p_value = np.nan
        srm_status = "NOT AVAILABLE"

    # ---------------------------------------------------------
    # 9. Guardrail assessment and telemetry coverage
    # ---------------------------------------------------------
    guardrail_df = _calculate_guardrails(
        exp_id=exp_id,
        assignments=a,
        guardrail_definitions=guardrails,
        guardrail_observations=guardrail_observations,
    )

    guardrails_status = _overall_guardrail_status(
        guardrail_df
    )

    guardrails_pass = (
        guardrails_status == "PASS"
    )

    # ---------------------------------------------------------
    # 10. Metric version and calculation compatibility
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

        metric_version = selected_metric.get(
            "version"
        )

        metric_name = selected_metric.get(
            "metric_name",
            "conversion_rate",
        )

    metric_supported, metric_definition_status = (
        _validate_itt_metric_definition(
            selected_metric
        )
    )

    # ---------------------------------------------------------
    # 11. Evidence-based decision
    # ---------------------------------------------------------
    significant = (
        np.isfinite(p_value)
        and p_value < alpha
    )

    if srm_status == "FAIL":
        decision = (
            "DO NOT ROLL OUT - SAMPLE RATIO MISMATCH"
        )

    elif not metric_supported:
        decision = (
            "INCONCLUSIVE - METRIC DEFINITION UNSUPPORTED"
        )

    elif guardrails_status == "FAIL":
        decision = (
            "DO NOT ROLL OUT - GUARDRAIL FAILURE"
        )

    elif guardrails_status != "PASS":
        decision = (
            "INCONCLUSIVE - GUARDRAIL EVIDENCE INSUFFICIENT"
        )

    elif significant and uplift > 0:
        decision = "PROMISING - REVIEW"

    elif significant and uplift < 0:
        decision = "TREATMENT UNDERPERFORMS"

    elif not np.isfinite(p_value):
        decision = "INCONCLUSIVE - INSUFFICIENT SAMPLE"

    else:
        decision = "INCONCLUSIVE"

    # ---------------------------------------------------------
    # 12. Return results
    # ---------------------------------------------------------
    return {
        "variants": out,
        "uplift_pct": (
            float(uplift)
            if np.isfinite(uplift)
            else np.nan
        ),
        "p_value": (
            float(p_value)
            if np.isfinite(p_value)
            else np.nan
        ),
        "ci_low": float(difference_ci[0]),
        "ci_high": float(difference_ci[1]),
        "srm_p_value": (
            float(srm_p_value)
            if np.isfinite(srm_p_value)
            else np.nan
        ),
        "srm_status": srm_status,
        "expected_allocation_a": expected_a,
        "expected_allocation_b": expected_b,
        "guardrails": guardrail_df,
        "guardrails_status": guardrails_status,
        "guardrails_pass": guardrails_pass,
        "assigned_total": int(total),
        "decision": decision,
        "excluded_conflict_assignments": int(
            excluded_conflict_assignments
        ),
        "conflicting_assignment_count": int(
            conflicting_assignment_count
        ),
        "conflicting_users": int(
            len(conflicting_users)
        ),
        "ineligible_assignment_count": int(
            ineligible_assignment_count
        ),
        "valid_conversion_events": int(
            conversion_quality["valid_conversion_events"]
        ),
        "duplicate_conversion_events": int(
            conversion_quality["duplicate_conversion_events"]
        ),
        "invalid_conversion_events": int(
            conversion_quality["invalid_conversion_events"]
        ),
        "mismatched_conversion_events": int(
            conversion_quality["mismatched_conversion_events"]
        ),
        "orphan_conversion_events": int(
            conversion_quality["orphan_conversion_events"]
        ),
        "analysis_window_days": ANALYSIS_WINDOW_DAYS,
        "analysis_at": analysis_at,
        "metric_name": metric_name,
        "metric_definition_id": metric_definition_id,
        "metric_version": metric_version,
        "metric_definition_status": metric_definition_status,
        "metric_supported": bool(metric_supported),

        # Original statistical metric names
        **power_metrics,

        # Compatibility aliases for dashboard and regression tests
        "estimated_power": power_metrics["statistical_power"],
        "mde_relative": power_metrics["mde_pct"],
        "sample_sufficient": power_metrics["sample_size_sufficient"],
    }


# =============================================================
# 6. Experiment summaries
# =============================================================

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
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "srm_p_value": result["srm_p_value"],
            "srm_status": result["srm_status"],
            "guardrails_status": result["guardrails_status"],
            "excluded_conflict_assignments": (
                result["excluded_conflict_assignments"]
            ),
            "ineligible_assignment_count": (
                result["ineligible_assignment_count"]
            ),
            "statistical_power": result["statistical_power"],
            "mde_absolute": result["mde_absolute"],
            "mde_pct": result["mde_pct"],
            "required_sample_size_per_arm": (
                result["required_sample_size_per_arm"]
            ),
            "sample_size_sufficient": (
                result["sample_size_sufficient"]
            ),
            "decision": result["decision"],
        })

    return pd.DataFrame(rows)


# =============================================================
# 7. Immutable result snapshots
# =============================================================

def build_result_records(
    frames,
    generated_at=DEFAULT_ANALYSIS_AT,
):
    """Build one result snapshot per experiment."""
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

        arms = result["variants"].set_index(
            "variant"
        )

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
