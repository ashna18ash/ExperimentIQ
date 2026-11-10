import pandas as pd

from src.generator import generate_dataset
from src.validation import validate_frames, integrity_summary
from src.analytics import analyze_experiment


def test_generator_small_dataset():
    f = generate_dataset(1000, 10, 20261022)
    assert len(f["experiments"]) == 10
    assert len(f["audience_members"]) == 1000
    assert f["ground_truth"].issue_type.nunique() == 5
    assert len(f["experiment_results"]) == 10


def test_deterministic_generation():
    a = generate_dataset(300, 10, 42)
    b = generate_dataset(300, 10, 42)
    assert a["audience_members"].equals(b["audience_members"])
    assert a["assignments"].equals(b["assignments"])
    assert a["experiment_results"].equals(b["experiment_results"])


def test_validation_finds_all_five_required_scenarios():
    f = generate_dataset(500, 10, 20261022)
    issues = validate_frames(f)
    found = set(issues.issue_type)
    expected = set(f["ground_truth"].issue_type)
    assert expected.issubset(found)


def test_integrity_checks_include_foreign_keys():
    f = generate_dataset(500, 10, 20261022)
    checks = integrity_summary(f)
    assert not checks.empty
    assert (checks.status == "PASS").all()
    assert checks.check.str.contains("foreign key").any()


def test_analysis_reports_uncertainty_and_configured_srm():
    f = generate_dataset(1000, 10, 20261022)
    exp = f["experiments"].experiment_id.iloc[0]
    r = analyze_experiment(exp, f["assignments"], f["conversion_events"], f["exposure_events"],
                           f["variants"], f["guardrail_metrics"], f["experiments"])
    assert 0 <= r["p_value"] <= 1
    assert r["ci_low"] <= r["ci_high"]
    assert r["expected_allocation_a"] == 0.5


def test_srm_scenario_is_flagged_against_expected_ratio():
    f = generate_dataset(1000, 10, 20261022)
    issues = validate_frames(f)
    assert "SAMPLE_RATIO_MISMATCH" in set(issues.issue_type)


def test_sqlite_roundtrip_enforces_relational_schema(tmp_path):
    import sqlite3
    from src.storage import save_frames, load_frames
    f = generate_dataset(300, 10, 20261022)
    db = tmp_path / "experimentiq.db"
    save_frames(f, db)
    loaded = load_frames(db)
    assert len(loaded["experiment_results"]) == 10
    assert len(loaded["ground_truth"]) == 5
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA foreign_keys = ON")
        foreign_keys = con.execute("PRAGMA foreign_key_list(assignments)").fetchall()
        assert len(foreign_keys) == 3
        try:
            con.execute("INSERT INTO variants (variant_id, experiment_id) VALUES ('BAD','NO-EXPERIMENT')")
            con.commit()
            assert False, "Foreign key should reject an unknown experiment"
        except sqlite3.IntegrityError:
            con.rollback()


def test_result_snapshots_are_retained_when_dataset_changes(tmp_path):
    from src.storage import save_frames, load_frames
    db = tmp_path / "history.db"
    first = generate_dataset(500, 10, 11)
    save_frames(first, db)
    second = generate_dataset(500, 10, 12)
    save_frames(second, db)
    stored = load_frames(db)["experiment_results"]
    assert len(stored) == 20
    assert stored.groupby("experiment_id").size().eq(2).all()
    assert stored.experiment_result_id.str.endswith(("V1", "V2")).all()



def test_out_of_window_conversion_does_not_change_primary_metric():
    from src.analytics import analyze_experiment
    f = generate_dataset(1000, 10, 20261022)
    exp = f["experiments"].experiment_id.iloc[0]
    before = analyze_experiment(exp, f["assignments"], f["conversion_events"], f["exposure_events"],
                                f["variants"], f["guardrail_metrics"], f["experiments"])
    assignment = f["assignments"].query("experiment_id == @exp").iloc[0]
    extra = {"conversion_id": "CONV-TEST-OUTSIDE", "assignment_id": assignment.assignment_id,
             "experiment_id": exp, "user_id": assignment.user_id,
             "converted_at": (pd.Timestamp(assignment.assigned_at) + pd.Timedelta(days=60)).isoformat(),
             "conversion_value": 5.0, "event_type": "purchase"}
    conversions = pd.concat([f["conversion_events"], pd.DataFrame([extra])], ignore_index=True)
    after = analyze_experiment(exp, f["assignments"], conversions, f["exposure_events"],
                               f["variants"], f["guardrail_metrics"], f["experiments"])
    assert before["variants"].conversions.tolist() == after["variants"].conversions.tolist()


def test_result_history_distinguishes_audience_scale(tmp_path):
    from src.storage import save_frames, load_frames
    from src.validation import integrity_summary
    db = tmp_path / "scale-history.db"
    small = generate_dataset(300, 10, 20261022)
    save_frames(small, db)
    full = generate_dataset(600, 10, 20261022)
    save_frames(full, db)
    stored = load_frames(db)
    assert len(stored["experiment_results"]) == 20
    checks = integrity_summary(stored)
    assert checks.loc[checks.check.str.contains("assigned-count reconciliation"), "status"].iloc[0] == "PASS"
