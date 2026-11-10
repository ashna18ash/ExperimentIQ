
"""SQLite persistence with declared primary and foreign-key constraints."""

import sqlite3
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import re

import pandas as pd
from pandas.api import types as ptypes


# ---------------------------------------------------------
# TABLE CONFIGURATION
# ---------------------------------------------------------

TABLES = [
    "experiments",
    "variants",
    "audience_members",
    "assignments",
    "exposure_events",
    "conversion_events",
    "metric_definitions",
    "guardrail_metrics",
    "guardrail_observations",
    "experiment_results"
]

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
    "experiment_results": "experiment_result_id"
}

FOREIGN_KEYS = {
    "variants": [
        ("experiment_id", "experiments", "experiment_id")
    ],

    "assignments": [
        ("experiment_id", "experiments", "experiment_id"),
        ("user_id", "audience_members", "user_id"),
        ("variant_id", "variants", "variant_id")
    ],

    "exposure_events": [
        ("assignment_id", "assignments", "assignment_id"),
        ("experiment_id", "experiments", "experiment_id"),
        ("user_id", "audience_members", "user_id"),
        ("variant_id", "variants", "variant_id")
    ],

    "conversion_events": [
        ("assignment_id", "assignments", "assignment_id"),
        ("experiment_id", "experiments", "experiment_id"),
        ("user_id", "audience_members", "user_id")
    ],

    "metric_definitions": [
        ("experiment_id", "experiments", "experiment_id")
    ],

    "guardrail_metrics": [
        ("experiment_id", "experiments", "experiment_id")
    ],

    "guardrail_observations": [
        ("assignment_id", "assignments", "assignment_id"),
        ("experiment_id", "experiments", "experiment_id"),
        ("user_id", "audience_members", "user_id")
    ],

    "experiment_results": [
        ("experiment_id", "experiments", "experiment_id")
    ]
}


# ---------------------------------------------------------
# SQLITE HELPERS
# ---------------------------------------------------------

def _sql_type(series):
    if ptypes.is_bool_dtype(series) or ptypes.is_integer_dtype(series):
        return "INTEGER"

    if ptypes.is_float_dtype(series):
        return "REAL"

    return "TEXT"


def _sql_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def _quote_identifier(name):
    """Safely quote an SQLite column or table identifier."""
    return '"' + str(name).replace('"', '""') + '"'


# ---------------------------------------------------------
# SAVE DATAFRAMES
# ---------------------------------------------------------

def save_frames(frames, db_path):
    """Save generated data with relational constraints and result history."""

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Ensure every configured table is available.
    missing = [
        table for table in TABLES
        if table not in frames
    ]

    if missing:
        raise ValueError(
            f"Missing required dataframes: {missing}"
        )

    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")

        # Preserve previous result snapshots.
        previous_results = pd.DataFrame()

        exists = con.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='experiment_results'"
        ).fetchone()

        if exists:
            previous_results = pd.read_sql_query(
                'SELECT * FROM "experiment_results"',
                con
            )

        incoming_results = frames["experiment_results"].copy()

        if not previous_results.empty:

            current_experiments = set(
                frames["experiments"].experiment_id.astype(str)
            )

            previous_results = previous_results[
                previous_results.experiment_id.astype(str).isin(
                    current_experiments
                )
            ]

            combined = previous_results.copy()
            now = datetime.now(timezone.utc).isoformat()

            for _, new_row in incoming_results.iterrows():

                old_rows = previous_results[
                    previous_results.experiment_id
                    == new_row.experiment_id
                ]

                compare_cols = [
                    c for c in incoming_results.columns
                    if c not in (
                        "experiment_result_id",
                        "generated_at"
                    )
                ]

                identical = False

                for _, old_row in old_rows.iterrows():

                    if all(
                        old_row.get(c) == new_row.get(c)
                        for c in compare_cols
                    ):
                        identical = True
                        break

                if identical:
                    continue

                versions = []

                for value in old_rows.experiment_result_id.astype(str):
                    match = re.search(r"-V(\d+)$", value)

                    if match:
                        versions.append(int(match.group(1)))

                next_version = max(versions, default=0) + 1

                new_row = new_row.copy()

                new_row["experiment_result_id"] = (
                    f"RES-{new_row.experiment_id}-V{next_version}"
                )

                new_row["generated_at"] = now

                combined = pd.concat(
                    [combined, new_row.to_frame().T],
                    ignore_index=True
                )

            frames = dict(frames)
            frames["experiment_results"] = combined

        # -------------------------------------------------
        # CREATE TABLES AND INSERT RECORDS TRANSACTIONALLY
        # -------------------------------------------------

        con.execute("BEGIN")

        try:

            # Drop child tables before parent tables.
            for table in reversed(TABLES):
                con.execute(
                    f"DROP TABLE IF EXISTS {_quote_identifier(table)}"
                )

            con.execute(
                'DROP TABLE IF EXISTS "ground_truth"'
            )

            # Create parent tables before dependent tables.
            for table in TABLES:

                df = frames[table]

                pk = PRIMARY_KEYS[table]

                if pk not in df.columns:
                    raise ValueError(
                        f"Primary key column {pk} missing from {table}"
                    )

                defs = [
                    f"{_quote_identifier(col)} {_sql_type(df[col])}"
                    for col in df.columns
                ]

                defs.append(
                    f"PRIMARY KEY ({_quote_identifier(pk)})"
                )

                for child_col, parent_table, parent_col in (
                    FOREIGN_KEYS.get(table, [])
                ):

                    if child_col not in df.columns:
                        raise ValueError(
                            f"Foreign key column {child_col} "
                            f"missing from {table}"
                        )

                    defs.append(
                        f"FOREIGN KEY ({_quote_identifier(child_col)}) "
                        f"REFERENCES {_quote_identifier(parent_table)}"
                        f"({_quote_identifier(parent_col)})"
                    )

                con.execute(
                    f"CREATE TABLE {_quote_identifier(table)} "
                    f"({', '.join(defs)})"
                )

            # Insert all analytical tables.
            for table in TABLES:

                df = frames[table]
                cols = list(df.columns)

                col_sql = ", ".join(
                    _quote_identifier(c) for c in cols
                )

                placeholders = ", ".join(
                    "?" for _ in cols
                )

                sql = (
                    f"INSERT INTO {_quote_identifier(table)} "
                    f"({col_sql}) VALUES ({placeholders})"
                )

                values = [
                    tuple(_sql_value(v) for v in row)
                    for row in df.itertuples(
                        index=False,
                        name=None
                    )
                ]

                con.executemany(sql, values)

            # Keep ground-truth evaluation labels separate.
            gt = frames["ground_truth"]

            gt_defs = [
                f"{_quote_identifier(col)} {_sql_type(gt[col])}"
                for col in gt.columns
            ]

            con.execute(
                f'CREATE TABLE "ground_truth" '
                f'({", ".join(gt_defs)})'
            )

            if not gt.empty:

                cols = list(gt.columns)

                col_sql = ", ".join(
                    _quote_identifier(c) for c in cols
                )

                placeholders = ", ".join(
                    "?" for _ in cols
                )

                sql = (
                    f'INSERT INTO "ground_truth" '
                    f"({col_sql}) VALUES ({placeholders})"
                )

                values = [
                    tuple(_sql_value(v) for v in row)
                    for row in gt.itertuples(
                        index=False,
                        name=None
                    )
                ]

                con.executemany(sql, values)

            con.commit()

        except Exception:
            con.rollback()
            raise


# ---------------------------------------------------------
# LOAD DATAFRAMES
# ---------------------------------------------------------

def load_frames(db_path):
    """Load all analytical tables and separate ground-truth labels."""

    with sqlite3.connect(db_path) as con:

        return {
            table: pd.read_sql_query(
                f"SELECT * FROM {_quote_identifier(table)}",
                con
            )
            for table in TABLES + ["ground_truth"]
        }


# ---------------------------------------------------------
# EXPORT CSV, JSON AND SQL BUNDLE
# ---------------------------------------------------------


def export_bundle(frames, output_dir):
    """Export portable CSV, JSON and dependency-ordered SQL datasets."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": int(frames["experiments"].seed.iloc[0]),
        "tables": {}
    }

    # -----------------------------------------------------
    # 1. EXPORT CSV AND JSON
    # -----------------------------------------------------

    for table, df in frames.items():

        csv_path = out / f"{table}.csv"
        json_path = out / f"{table}.json"

        df.to_csv(csv_path, index=False)

        df.to_json(
            json_path,
            orient="records",
            indent=2,
            date_format="iso"
        )

        manifest["tables"][table] = {
            "rows": int(len(df)),
            "csv_sha256": hashlib.sha256(
                csv_path.read_bytes()
            ).hexdigest(),
            "json_sha256": hashlib.sha256(
                json_path.read_bytes()
            ).hexdigest()
        }

    # -----------------------------------------------------
    # 2. CREATE TEMPORARY RELATIONAL DATABASE
    # -----------------------------------------------------

    sql_db = out / ".experimentiq_export_tmp.db"

    if sql_db.exists():
        sql_db.unlink()

    save_frames(frames, sql_db)

    con = sqlite3.connect(sql_db)

    try:

        # Get CREATE TABLE statements separately.
        schema = {}

        rows = con.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table'
              AND sql IS NOT NULL
            """
        ).fetchall()

        for name, statement in rows:
            schema[name] = statement + ";"

        # Preserve the declared parent-to-child order.
        export_order = TABLES + ["ground_truth"]

        missing_schema = [
            table for table in export_order
            if table not in schema
        ]

        if missing_schema:
            raise ValueError(
                f"SQL export missing table definitions: {missing_schema}"
            )

        # Get INSERT statements from SQLite.
        insert_statements = list(con.iterdump())

        inserts_by_table = {
            table: [] for table in export_order
        }

        for statement in insert_statements:

            if not statement.startswith("INSERT INTO "):
                continue

            match = re.match(
                r'INSERT INTO "([^"]+)"',
                statement
            )

            if not match:
                continue

            table_name = match.group(1)

            if table_name in inserts_by_table:
                inserts_by_table[table_name].append(
                    statement
                )

        # -------------------------------------------------
        # 3. BUILD ORDERED, PORTABLE SQL SCRIPT
        # -------------------------------------------------

        sql_lines = [
            "-- ExperimentIQ portable database export",
            "-- Generated from the synthetic source dataset",
            "-- Primary and foreign-key constraints are retained.",
            "",
            "PRAGMA foreign_keys = ON;",
            "BEGIN TRANSACTION;",
            ""
        ]

        # First create EVERY table.
        for table in export_order:
            sql_lines.append(schema[table])
            sql_lines.append("")

        # Then insert records in parent-to-child order.
        for table in export_order:

            sql_lines.extend(
                inserts_by_table[table]
            )

            sql_lines.append("")

        sql_lines.extend([
            "COMMIT;",
            ""
        ])

        sql_text = "\n".join(sql_lines)

    finally:
        con.close()

    sql_path = out / "experimentiq_seed.sql"

    sql_path.write_text(
        sql_text,
        encoding="utf-8"
    )

    # Remove temporary SQLite database.
    sql_db.unlink(missing_ok=True)

    # -----------------------------------------------------
    # 4. UPDATE SQL CHECKSUM
    # -----------------------------------------------------

    manifest["sql_seed_sha256"] = hashlib.sha256(
        sql_path.read_bytes()
    ).hexdigest()

    # -----------------------------------------------------
    # 5. INCLUDE PROJECT DOCUMENTATION
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    for doc_name in [
        "README.md",
        "DATA_DICTIONARY.md",
        "SOLUTION_OVERVIEW.md"
    ]:

        source = project_root / doc_name

        if source.exists():
            (out / doc_name).write_bytes(
                source.read_bytes()
            )

    # -----------------------------------------------------
    # 6. SAVE SOURCE MANIFEST
    # -----------------------------------------------------

    (out / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    return manifest
