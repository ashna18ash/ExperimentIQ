
import numpy as np
import pandas as pd

from src.generator import generate_dataset
from src.validation import (
    validate_frames,
    integrity_summary,
    severity_summary,
)
from src.analytics import (
    analyze_experiment,
    wilson_interval,
    newcombe_difference_interval,
)


# =============================================================
# 1. DATASET GENERATION
# =============================================================

def test_generator_small_dataset():
    frames = generate_dataset(1000, 10, 20261022)

    assert len(frames["experiments"]) == 10
    assert len(frames["audience_members"]) == 1000
    assert frames["ground_truth"].issue_type.nunique() == 5
    assert len(frames["experiment_results"]) == 10


def test_deterministic_generation():
    first = generate_dataset(300, 10, 42)
    second = generate_dataset(300, 10, 42)

    assert first["audience_members"].equals(
        second["audience_members"]
    )

    assert first["assignments"].equals(
        second["assignments"]
    )

    assert first["experiment_results"].equals(
        second["experiment_results"]
    )


# =============================================================
# 2. DATA VALIDATION
# =============================================================

def test_validation_finds_all_five_required_scenarios():
    frames = generate_dataset(500, 10, 20261022)

    issues = validate_frames(frames)

    found = set(issues.issue_type)
    expected = set(frames["ground_truth"].issue_type)

    assert expected.issubset(found)


def test_integrity_checks_include_foreign_keys():
    frames = generate_dataset(500, 10, 20261022)

    checks = integrity_summary(frames)

    assert not checks.empty
    assert (checks.status == "PASS").all()
    assert checks.check.str.contains("foreign key").any()


def test_srm_scenario_is_flagged_against_expected_ratio():
    frames = generate_dataset(1000, 10, 20261022)

    issues = validate_frames(frames)

    assert "SAMPLE_RATIO_MISMATCH" in set(
        issues.issue_type
    )


# =============================================================
# 3. ANALYTICS AND STATISTICAL TESTING
# =============================================================

def test_analysis_reports_uncertainty_and_configured_srm():
    frames = generate_dataset(1000, 10, 20261022)

    experiment_id = (
        frames["experiments"]
        .experiment_id.iloc[0]
    )

    result = analyze_experiment(
        experiment_id,
        frames["assignments"],
        frames["conversion_events"],
        frames["exposure_events"],
        frames["variants"],
        frames["guardrail_metrics"],
        frames["experiments"],
    )

    assert 0 <= result["p_value"] <= 1
    assert result["ci_low"] <= result["ci_high"]
    assert result["expected_allocation_a"] == 0.5


def test_wilson_interval_is_valid():
    lower, upper = wilson_interval(
        successes=50,
        total=100,
    )

    assert np.isfinite(lower)
    assert np.isfinite(upper)

    assert 0 <= lower <= upper <= 1


def test_wilson_interval_handles_zero_sample():
    lower, upper = wilson_interval(
        successes=0,
        total=0,
    )

    assert np.isnan(lower)
    assert np.isnan(upper)


def test_newcombe_difference_interval_is_ordered():
    lower, upper = newcombe_difference_interval(
        successes_a=50,
        total_a=100,
        successes_b=60,
        total_b=100,
    )

    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert lower <= upper

    # The observed difference is 10 percentage points.
    assert lower < 0.10 < upper


def test_newcombe_interval_detects_clear_difference():
    lower, upper = newcombe_difference_interval(
        successes_a=20,
        total_a=1000,
        successes_b=100,
        total_b=1000,
    )

    assert lower < upper
    assert lower > 0


def test_analysis_reports_power_and_mde_metrics():
    frames = generate_dataset(10000, 10, 20261022)

    experiment_id = frames["experiments"].iloc[0][
        "experiment_id"
    ]

    result = analyze_experiment(
        experiment_id,
        frames["assignments"],
        frames["conversion_events"],
        frames["exposure_events"],
        frames["variants"],
        frames["guardrail_metrics"],
        experiments=frames["experiments"],
        guardrail_observations=frames[
            "guardrail_observations"
        ],
        metric_definitions=frames["metric_definitions"],
    )

    assert "estimated_power" in result
    assert "mde_absolute" in result
    assert "mde_relative" in result
    assert "required_sample_size_per_arm" in result
    assert "sample_sufficient" in result

    assert 0 <= result["estimated_power"] <= 1

    assert (
        result["required_sample_size_per_arm"] > 0
    )

    assert result["mde_absolute"] >= 0


# =============================================================
# 4. ALLOCATION AND ASSIGNMENT TESTS
# =============================================================

def test_unbalanced_40_60_allocation_is_not_flagged_as_srm():
    frames = generate_dataset(10000, 10, 20261022)

    experiment_id = frames["experiments"].iloc[1][
        "experiment_id"
    ]

    result = analyze_experiment(
        experiment_id,
        frames["assignments"],
        frames["conversion_events"],
        frames["exposure_events"],
        frames["variants"],
        frames["guardrail_metrics"],
        experiments=frames["experiments"],
        guardrail_observations=frames[
            "guardrail_observations"
        ],
        metric_definitions=frames["metric_definitions"],
    )

    assert result["expected_allocation_a"] == 0.40
    assert result["expected_allocation_b"] == 0.60
    assert result["srm_status"] == "PASS"


# =============================================================
# 5. GUARDRAIL TESTS
# =============================================================

def test_guardrail_regression_is_detected_from_observations():
    frames = generate_dataset(10000, 10, 20261022)

    experiment_id = frames["experiments"].iloc[4][
        "experiment_id"
    ]

    assignments = frames["assignments"].merge(
        frames["variants"][
            ["variant_id", "variant_key"]
        ],
        on="variant_id",
        how="left",
    )

    experiment_assignments = assignments[
        assignments["experiment_id"] == experiment_id
    ]

    treatment_ids = set(
        experiment_assignments.loc[
            experiment_assignments["variant_key"] == "B",
            "assignment_id",
        ]
    )

    control_ids = set(
        experiment_assignments.loc[
            experiment_assignments["variant_key"] == "A",
            "assignment_id",
        ]
    )

    observations = frames[
        "guardrail_observations"
    ].copy()

    treatment_mask = observations["assignment_id"].isin(
        treatment_ids
    )

    control_mask = observations["assignment_id"].isin(
        control_ids
    )

    observations.loc[
        treatment_mask,
        ["error_occurred", "refund_occurred"],
    ] = True

    observations.loc[
        control_mask,
        ["error_occurred", "refund_occurred"],
    ] = False

    result = analyze_experiment(
        experiment_id,
        frames["assignments"],
        frames["conversion_events"],
        frames["exposure_events"],
        frames["variants"],
        frames["guardrail_metrics"],
        experiments=frames["experiments"],
        guardrail_observations=observations,
        metric_definitions=frames["metric_definitions"],
    )

    assert not result["guardrails"].empty

    assert set(
        result["guardrails"]["status"]
    ) == {"FAIL"}

    assert (
        result["decision"]
        == "DO NOT ROLL OUT - GUARDRAIL FAILURE"
    )


def test_missing_guardrail_observations_do_not_pass():
    frames = generate_dataset(1000, 10, 20261022)

    experiment_id = frames["experiments"].iloc[0][
        "experiment_id"
    ]

    # Attach variant information so the same population used by
    # analyze_experiment can be identified.
    assignments = frames["assignments"].copy()

    assignments = assignments.merge(
        frames["variants"][
            ["variant_id", "variant_key"]
        ],
        on="variant_id",
        how="left",
    )

    experiment_assignments = assignments.loc[
        assignments["experiment_id"] == experiment_id
    ].copy()

    # Apply the same eligibility filtering as the ITT analysis.
    eligibility_column = next(
        (
            column
            for column in ["is_eligible", "eligible"]
            if column in experiment_assignments.columns
        ),
        None,
    )

    if eligibility_column:
        experiment_assignments = experiment_assignments.loc[
            experiment_assignments[eligibility_column]
            .fillna(False)
            .astype(bool)
        ].copy()

    # Exclude users assigned to more than one variant.
    variants_per_user = (
        experiment_assignments.groupby("user_id")[
            "variant_id"
        ].nunique()
    )

    conflicting_users = set(
        variants_per_user[
            variants_per_user > 1
        ].index
    )

    experiment_assignments = experiment_assignments.loc[
        ~experiment_assignments["user_id"].isin(
            conflicting_users
        )
    ].copy()

    assert not experiment_assignments.empty

    # Remove an observation belonging to an assignment that is
    # actually retained by the ITT analysis.
    missing_assignment_id = experiment_assignments.iloc[0][
        "assignment_id"
    ]

    observations = frames[
        "guardrail_observations"
    ].copy()

    observations = observations.loc[
        observations["assignment_id"]
        != missing_assignment_id
    ].copy()

    result = analyze_experiment(
        experiment_id,
        frames["assignments"],
        frames["conversion_events"],
        frames["exposure_events"],
        frames["variants"],
        frames["guardrail_metrics"],
        experiments=frames["experiments"],
        guardrail_observations=observations,
        metric_definitions=frames["metric_definitions"],
    )

    assert not result["guardrails"].empty

    assert (
        "INSUFFICIENT DATA"
        in set(result["guardrails"]["status"])
    )

    assert result["guardrails_status"] != "PASS"


# =============================================================
# 6. METRIC VERSIONING
# =============================================================

def test_metric_definition_uses_effective_version_at_analysis_time():
    frames = generate_dataset(1000, 10, 20261022)

    experiment_id = frames["experiments"].iloc[0][
        "experiment_id"
    ]

    common_args = (
        experiment_id,
        frames["assignments"],
        frames["conversion_events"],
        frames["exposure_events"],
        frames["variants"],
        frames["guardrail_metrics"],
    )

    before_version_2 = analyze_experiment(
        *common_args,
        experiments=frames["experiments"],
        guardrail_observations=frames[
            "guardrail_observations"
        ],
        metric_definitions=frames["metric_definitions"],
        analysis_at="2026-10-31T23:59:00+00:00",
    )

    at_version_2 = analyze_experiment(
        *common_args,
        experiments=frames["experiments"],
        guardrail_observations=frames[
            "guardrail_observations"
        ],
        metric_definitions=frames["metric_definitions"],
        analysis_at="2026-11-01T00:00:00+00:00",
    )

    assert before_version_2["metric_version"] == 1
    assert at_version_2["metric_version"] == 2

    assert (
        before_version_2["metric_definition_id"]
        != at_version_2["metric_definition_id"]
    )


# =============================================================
# 7. CONVERSION DATA-QUALITY TESTS
# =============================================================

def test_out_of_window_conversion_does_not_change_primary_metric():
    frames = generate_dataset(1000, 10, 20261022)

    experiment_id = frames["experiments"].iloc[0][
        "experiment_id"
    ]

    before = analyze_experiment(
        experiment_id,
        frames["assignments"],
        frames["conversion_events"],
        frames["exposure_events"],
        frames["variants"],
        frames["guardrail_metrics"],
        frames["experiments"],
    )

    assignment = frames["assignments"].query(
        "experiment_id == @experiment_id"
    ).iloc[0]

    extra = {
        "conversion_id": "CONV-TEST-OUTSIDE",
        "assignment_id": assignment.assignment_id,
        "experiment_id": experiment_id,
        "user_id": assignment.user_id,
        "converted_at": (
            pd.Timestamp(assignment.assigned_at)
            + pd.Timedelta(days=60)
        ).isoformat(),
        "conversion_value": 5.0,
        "event_type": "purchase",
    }

    conversions = pd.concat(
        [
            frames["conversion_events"],
            pd.DataFrame([extra]),
        ],
        ignore_index=True,
    )

    after = analyze_experiment(
        experiment_id,
        frames["assignments"],
        conversions,
        frames["exposure_events"],
        frames["variants"],
        frames["guardrail_metrics"],
        frames["experiments"],
    )

    assert (
        before["variants"]["conversions"].tolist()
        == after["variants"]["conversions"].tolist()
    )


def test_duplicate_conversion_is_detected():
    frames = generate_dataset(500, 10, 20261022)

    conversions = frames["conversion_events"].copy()

    duplicate = conversions.iloc[[0]].copy()

    conversions = pd.concat(
        [conversions, duplicate],
        ignore_index=True,
    )

    modified = dict(frames)
    modified["conversion_events"] = conversions

    issues = validate_frames(modified)

    assert "DUPLICATE_PRIMARY_KEY" in set(
        issues["issue_type"]
    )


def test_orphan_conversion_assignment_is_detected():
    frames = generate_dataset(500, 10, 20261022)

    conversions = frames["conversion_events"].copy()

    orphan = conversions.iloc[[0]].copy()
    orphan["conversion_id"] = "CONV-ORPHAN-TEST"
    orphan["assignment_id"] = "ASSIGNMENT-DOES-NOT-EXIST"

    modified = dict(frames)

    modified["conversion_events"] = pd.concat(
        [conversions, orphan],
        ignore_index=True,
    )

    issues = validate_frames(modified)

    assert "ORPHAN_ASSIGNMENT" in set(
        issues["issue_type"]
    )


def test_invalid_conversion_timestamp_is_detected():
    frames = generate_dataset(500, 10, 20261022)

    conversions = frames["conversion_events"].copy()

    conversions.loc[
        conversions.index[0],
        "converted_at",
    ] = "not-a-valid-timestamp"

    modified = dict(frames)
    modified["conversion_events"] = conversions

    issues = validate_frames(modified)

    assert "INVALID_EVENT_TIMESTAMP" in set(
        issues["issue_type"]
    )


# =============================================================
# 8. SEVERITY SUMMARY TESTS
# =============================================================

def test_severity_summary_counts_all_levels():
    issues = pd.DataFrame({
        "severity": [
            "HIGH",
            "HIGH",
            "MEDIUM",
            "LOW",
            "CRITICAL",
        ],
        "issue_type": [
            "A",
            "B",
            "C",
            "D",
            "E",
        ],
    })

    summary = severity_summary(issues)

    counts = dict(
        zip(
            summary["severity"],
            summary["issue_count"],
        )
    )

    assert counts["CRITICAL"] == 1
    assert counts["HIGH"] == 2
    assert counts["MEDIUM"] == 1
    assert counts["LOW"] == 1


def test_severity_summary_handles_empty_issues():
    summary = severity_summary(
        pd.DataFrame()
    )

    assert len(summary) == 4
    assert summary["issue_count"].sum() == 0


# =============================================================
# 9. SQLITE AND RESULT HISTORY
# =============================================================

def test_sqlite_roundtrip_enforces_relational_schema(tmp_path):
    import sqlite3

    from src.storage import save_frames, load_frames

    frames = generate_dataset(300, 10, 20261022)

    db = tmp_path / "experimentiq.db"

    save_frames(frames, db)

    loaded = load_frames(db)

    assert len(loaded["experiment_results"]) == 10
    assert len(loaded["ground_truth"]) == 5

    with sqlite3.connect(db) as connection:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(assignments)"
        ).fetchall()

        assert len(foreign_keys) == 3

        try:
            connection.execute(
                """
                INSERT INTO variants
                (variant_id, experiment_id)
                VALUES ('BAD', 'NO-EXPERIMENT')
                """
            )

            connection.commit()

            assert False, (
                "Foreign key should reject an unknown experiment"
            )

        except sqlite3.IntegrityError:
            connection.rollback()


def test_result_snapshots_are_retained_when_dataset_changes(tmp_path):
    from src.storage import save_frames, load_frames

    db = tmp_path / "history.db"

    first = generate_dataset(500, 10, 11)
    save_frames(first, db)

    second = generate_dataset(500, 10, 12)
    save_frames(second, db)

    stored = load_frames(db)["experiment_results"]

    assert len(stored) == 20

    assert (
        stored.groupby("experiment_id")
        .size()
        .eq(2)
        .all()
    )

    assert stored.experiment_result_id.str.endswith(
        ("V1", "V2")
    ).all()


def test_result_history_distinguishes_audience_scale(tmp_path):
    from src.storage import save_frames, load_frames

    db = tmp_path / "scale-history.db"

    small = generate_dataset(300, 10, 20261022)
    save_frames(small, db)

    full = generate_dataset(600, 10, 20261022)
    save_frames(full, db)

    stored = load_frames(db)

    assert len(stored["experiment_results"]) == 20

    checks = integrity_summary(stored)

    reconciliation = checks.loc[
        checks.check.str.contains(
            "assigned-count reconciliation"
        ),
        "status",
    ]

    assert not reconciliation.empty
    assert reconciliation.iloc[0] == "PASS"


# =============================================================
# 10. GROUND-TRUTH DECISION TESTS
# =============================================================

def test_ground_truth_expected_decisions_match_all_five_scenarios():
    frames = generate_dataset(1000, 10, 20261022)

    ground_truth = frames["ground_truth"]

    expected = {
        "MULTIPLE_VARIANT_ASSIGNMENT":
            "FLAG_CONFLICTING_ASSIGNMENT",

        "EXPOSURE_BEFORE_ASSIGNMENT":
            "FLAG_INVALID_EVENT_ORDER",

        "OUTCOME_WINDOW_INCONSISTENT":
            "FLAG_OUTSIDE_WINDOW",

        "METRIC_CHANGED_AFTER_RESULTS":
            "FLAG_VERSION_DRIFT",

        "SAMPLE_RATIO_MISMATCH":
            "FLAG_ALLOCATION_MISMATCH",
    }

    actual = dict(
        zip(
            ground_truth["issue_type"],
            ground_truth["expected_decision"],
        )
    )

    assert len(ground_truth) == 5
    assert ground_truth["issue_type"].nunique() == 5
    assert actual == expected
