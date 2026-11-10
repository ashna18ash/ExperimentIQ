
import numpy as np
import pandas as pd

from src.analytics import build_result_records

SEED_DEFAULT = 20261022


def _id(prefix, n):
    return f"{prefix}-2026-{n:06d}"


def generate_dataset(n_users=50000, n_experiments=10, seed=SEED_DEFAULT):
    """Generate deterministic synthetic experimentation data.

    Master data and definitions are generated before dependent records.
    Ground-truth anomaly labels remain separate from analytical inputs.
    """

    if n_experiments < 1:
        raise ValueError("n_experiments must be at least 1")

    if n_users < n_experiments * 2:
        raise ValueError("n_users must be at least twice n_experiments")

    rng = np.random.default_rng(seed)

    per_exp = [n_users // n_experiments] * n_experiments
    for i in range(n_users % n_experiments):
        per_exp[i] += 1

    experiments = []
    variants = []
    audience = []
    assignments = []

    exposures = []
    conversions = []
    metrics = []
    guardrails = []
    guardrail_observations = []

    truth = []

    user_counter = 0
    exposure_counter = 0
    conversion_counter = 0
    guardrail_observation_counter = 0

    experiment_names = [
        "Checkout Optimization",
        "Onboarding Redesign",
        "Search Ranking Test",
        "Email Subject Test",
        "Pricing Page Layout",
        "Recommendation Engine",
        "Signup Flow",
        "Notification Timing",
        "Cart Recovery",
        "Landing Page"
    ]

    # ---------------------------------------------------------
    # PHASE 1: GENERATE MASTER DATA AND EXPERIMENT DEFINITIONS
    # ---------------------------------------------------------

    for e in range(n_experiments):
        exp_id = _id("EXP", e + 1)

        start = (
            pd.Timestamp("2026-10-01", tz="UTC")
            + pd.Timedelta(days=e * 2)
        )

        # Experiment 2 intentionally uses a declared 40:60 split.
        allocation_b = (
            0.60 if e == 1
            else (0.50 if e % 2 == 0 else 0.52)
        )

        experiments.append({
            "experiment_id": exp_id,
            "name": experiment_names[e % len(experiment_names)],
            "hypothesis": (
                "The treatment variant improves conversion "
                "without unacceptable guardrail regression."
            ),
            "primary_metric": "conversion_rate",
            "status": "completed",
            "start_at": start.isoformat(),
            "end_at": (start + pd.Timedelta(days=14)).isoformat(),
            "allocation_a": 1.0 - allocation_b,
            "allocation_b": allocation_b,
            "seed": seed,
            "analysis_plan_version": "1.0"
        })

        variants.extend([
            {
                "variant_id": f"VAR-{exp_id}-A",
                "experiment_id": exp_id,
                "name": "Control",
                "variant_key": "A",
                "is_control": True
            },
            {
                "variant_id": f"VAR-{exp_id}-B",
                "experiment_id": exp_id,
                "name": "Treatment",
                "variant_key": "B",
                "is_control": False
            }
        ])

        metrics.append({
            "metric_definition_id": f"MET-{e + 1:06d}-V1",
            "experiment_id": exp_id,
            "metric_name": "conversion_rate",
            "numerator": "converted_assigned_users",
            "denominator": "assigned_users",
            "unit": "proportion",
            "version": 1,
            "effective_from": start.isoformat(),
            "effective_to": None
        })

        guardrails.extend([
            {
                "guardrail_id": f"GRD-{e + 1:06d}-01",
                "experiment_id": exp_id,
                "metric_name": "error_rate",
                "threshold": 0.08,
                "direction": "max",
                "unit": "proportion"
            },
            {
                "guardrail_id": f"GRD-{e + 1:06d}-02",
                "experiment_id": exp_id,
                "metric_name": "refund_rate",
                "threshold": 0.06,
                "direction": "max",
                "unit": "proportion"
            }
        ])

    # ---------------------------------------------------------
    # PHASE 2: GENERATE AUDIENCE MASTER DATA
    # ---------------------------------------------------------

    experiment_user_groups = []

    for e in range(n_experiments):
        exp_id = _id("EXP", e + 1)
        n = per_exp[e]

        start = (
            pd.Timestamp("2026-10-01", tz="UTC")
            + pd.Timedelta(days=e * 2)
        )

        user_ids = [
            f"USR-{user_counter + i + 1:07d}"
            for i in range(n)
        ]

        user_counter += n

        eligible = np.ones(n, dtype=bool)

        for i, uid in enumerate(user_ids):
            audience.append({
                "user_id": uid,
                "eligible": bool(eligible[i]),
                "segment": rng.choice([
                    "new",
                    "returning",
                    "high_value"
                ]),
                "country": rng.choice([
                    "IN",
                    "US",
                    "UK",
                    "CA"
                ]),
                "created_at": (
                    start
                    - pd.Timedelta(
                        days=int(rng.integers(1, 180))
                    )
                ).isoformat()
            })

        experiment_user_groups.append({
            "experiment_id": exp_id,
            "user_ids": user_ids,
            "eligible": eligible
        })

    # ---------------------------------------------------------
    # PHASE 3: GENERATE ASSIGNMENTS AND DEPENDENT EVENTS
    # ---------------------------------------------------------

    for e, group in enumerate(experiment_user_groups):
        exp_id = group["experiment_id"]
        user_ids = group["user_ids"]
        eligible = group["eligible"]

        start = (
            pd.Timestamp("2026-10-01", tz="UTC")
            + pd.Timedelta(days=e * 2)
        )

        allocation_b = (
            0.60 if e == 1
            else (0.50 if e % 2 == 0 else 0.52)
        )

        eligible_idx = np.where(eligible)[0]

        # Fixed-seed randomized assignment.
        arm_b = rng.random(len(eligible_idx)) < allocation_b

        for j, idx in enumerate(eligible_idx):
            uid = user_ids[idx]
            arm = "B" if arm_b[j] else "A"

            assigned_at = (
                start
                + pd.Timedelta(
                    minutes=int(rng.integers(0, 60 * 24 * 3))
                )
            )

            asg_id = f"ASG-{exp_id}-{uid}"
            variant_id = f"VAR-{exp_id}-{arm}"

            assignments.append({
                "assignment_id": asg_id,
                "experiment_id": exp_id,
                "user_id": uid,
                "variant_id": variant_id,
                "assigned_at": assigned_at.isoformat(),
                "assignment_status": "assigned",
                "is_eligible": True
            })

            # -------------------------------------------------
            # SOURCE-LEVEL GUARDRAIL OBSERVATIONS
            # -------------------------------------------------

            # Experiments 5 and 8 intentionally have treatment
            # guardrail regressions.
            treatment_regression = e in (4, 7) and arm == "B"

            error_probability = (
                0.105 if treatment_regression else 0.025
            )

            refund_probability = (
                0.075 if treatment_regression else 0.015
            )

            guardrail_observation_counter += 1

            guardrail_observations.append({
                "guardrail_observation_id": (
                    f"GRO-2026-{guardrail_observation_counter:06d}"
                ),
                "assignment_id": asg_id,
                "experiment_id": exp_id,
                "user_id": uid,
                "error_occurred": bool(
                    rng.random() < error_probability
                ),
                "refund_occurred": bool(
                    rng.random() < refund_probability
                ),
                "observed_at": (
                    assigned_at + pd.Timedelta(hours=1)
                ).isoformat()
            })

            # -------------------------------------------------
            # EXPOSURE EVENTS
            # -------------------------------------------------

            exposed = rng.random() > 0.07

            if exposed:
                exposure_counter += 1

                exposure_at = (
                    assigned_at
                    + pd.Timedelta(
                        minutes=int(rng.integers(1, 240))
                    )
                )

                missing_telemetry = rng.random() < 0.025

                exposures.append({
                    "exposure_id": _id("EXPOS", exposure_counter),
                    "assignment_id": asg_id,
                    "experiment_id": exp_id,
                    "user_id": uid,
                    "variant_id": variant_id,
                    "exposed_at": exposure_at.isoformat(),
                    "telemetry_status": (
                        "missing" if missing_telemetry
                        else "captured"
                    )
                })

            # -------------------------------------------------
            # CONVERSION EVENTS
            # -------------------------------------------------

            base = 0.105 + (
                0.008 if e % 3 == 0 else 0
            )

            lift = (
                0.025 if e not in (4, 7)
                else -0.005
            )

            p_conv = base + (
                lift if arm == "B" else 0
            )

            converted = rng.random() < p_conv

            if converted:
                conversion_counter += 1

                delay_hours = int(
                    rng.integers(1, 24 * 18)
                )

                conv_at = (
                    assigned_at
                    + pd.Timedelta(hours=delay_hours)
                )

                conversions.append({
                    "conversion_id": _id(
                        "CONV",
                        conversion_counter
                    ),
                    "assignment_id": asg_id,
                    "experiment_id": exp_id,
                    "user_id": uid,
                    "converted_at": conv_at.isoformat(),
                    "conversion_value": round(
                        float(rng.lognormal(3.4, 0.55)),
                        2
                    ),
                    "event_type": "purchase"
                })

    # ---------------------------------------------------------
    # PHASE 4: INJECT SEPARATELY LABELLED ANOMALIES
    # ---------------------------------------------------------

    if assignments:
        a0 = assignments[0]

        # Scenario 1: conflicting assignment for the same user.
        duplicate = dict(a0)

        duplicate["assignment_id"] = (
            a0["assignment_id"] + "-DUP"
        )

        duplicate["variant_id"] = (
            a0["variant_id"].replace("-A", "-B")
            if a0["variant_id"].endswith("-A")
            else a0["variant_id"].replace("-B", "-A")
        )

        assignments.append(duplicate)

        truth.append({
            "entity_type": "Assignment",
            "entity_id": duplicate["assignment_id"],
            "issue_type": "MULTIPLE_VARIANT_ASSIGNMENT",
            "expected_decision": "FLAG_CONFLICTING_ASSIGNMENT"
        })

        # Scenario 2: exposure before assignment.
        ex_id = _id("EXPOS", exposure_counter + 1)

        exposures.append({
            "exposure_id": ex_id,
            "assignment_id": a0["assignment_id"],
            "experiment_id": a0["experiment_id"],
            "user_id": a0["user_id"],
            "variant_id": a0["variant_id"],
            "exposed_at": (
                pd.Timestamp(a0["assigned_at"])
                - pd.Timedelta(hours=1)
            ).isoformat(),
            "telemetry_status": "captured"
        })

        truth.append({
            "entity_type": "ExposureEvent",
            "entity_id": ex_id,
            "issue_type": "EXPOSURE_BEFORE_ASSIGNMENT",
            "expected_decision": "FLAG_INVALID_EVENT_ORDER"
        })

        # Scenario 3: conversion outside the 30-day window.
        c_id = _id("CONV", conversion_counter + 1)

        conversions.append({
            "conversion_id": c_id,
            "assignment_id": a0["assignment_id"],
            "experiment_id": a0["experiment_id"],
            "user_id": a0["user_id"],
            "converted_at": (
                pd.Timestamp(a0["assigned_at"])
                + pd.Timedelta(days=60)
            ).isoformat(),
            "conversion_value": 25.0,
            "event_type": "purchase"
        })

        truth.append({
            "entity_type": "ConversionEvent",
            "entity_id": c_id,
            "issue_type": "OUTCOME_WINDOW_INCONSISTENT",
            "expected_decision": "FLAG_OUTSIDE_WINDOW"
        })

        # Scenario 4: metric definition changes after results.
        metrics.append({
            "metric_definition_id": "MET-000001-V2",
            "experiment_id": experiments[0]["experiment_id"],
            "metric_name": "conversion_rate",
            "numerator": "converted_exposed_users",
            "denominator": "exposed_users",
            "unit": "proportion",
            "version": 2,
            "effective_from": "2026-11-01T00:00:00+00:00",
            "effective_to": None
        })

        truth.append({
            "entity_type": "MetricDefinition",
            "entity_id": "MET-000001-V2",
            "issue_type": "METRIC_CHANGED_AFTER_RESULTS",
            "expected_decision": "FLAG_VERSION_DRIFT"
        })

        # Scenario 5: sample-ratio mismatch.
        # Experiment 3 declares 50:50 but its observed assignment
        # distribution is deliberately changed.
        srm_exp = experiments[2]["experiment_id"]

        srm_rows = [
            row for row in assignments
            if row["experiment_id"] == srm_exp
            and row["variant_id"].endswith("-B")
        ]

        flip_count = max(
            1,
            int(len(srm_rows) * 0.65)
        )

        flipped_ids = set()

        for row in srm_rows[:flip_count]:
            row["variant_id"] = f"VAR-{srm_exp}-A"
            flipped_ids.add(row["assignment_id"])

        for row in exposures:
            if row["assignment_id"] in flipped_ids:
                row["variant_id"] = f"VAR-{srm_exp}-A"

        truth.append({
            "entity_type": "Experiment",
            "entity_id": srm_exp,
            "issue_type": "SAMPLE_RATIO_MISMATCH",
            "expected_decision": "FLAG_ALLOCATION_MISMATCH"
        })

    # ---------------------------------------------------------
    # PHASE 5: BUILD DATAFRAMES
    # ---------------------------------------------------------

    frames = {
        "experiments": pd.DataFrame(experiments),
        "variants": pd.DataFrame(variants),
        "audience_members": pd.DataFrame(audience),
        "assignments": pd.DataFrame(assignments),
        "exposure_events": pd.DataFrame(exposures),
        "conversion_events": pd.DataFrame(conversions),
        "metric_definitions": pd.DataFrame(metrics),
        "guardrail_metrics": pd.DataFrame(guardrails),
        "guardrail_observations": pd.DataFrame(
            guardrail_observations
        ),
        "experiment_results": pd.DataFrame(),
        "ground_truth": pd.DataFrame(truth)
    }

    # Calculate result snapshots from generated source records.
    frames["experiment_results"] = build_result_records(frames)

    return frames
