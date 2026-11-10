"""ExperimentIQ — Experimentation & Decision Intelligence dashboard."""

import sys
import io
import zipfile
import hashlib
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import DB_PATH, SEED
from src.generator import generate_dataset
from src.storage import save_frames, load_frames, export_bundle
from src.analytics import analyze_experiment, all_experiment_summaries
from src.validation import validate_frames, integrity_summary


# =============================================================
# PAGE CONFIGURATION
# =============================================================

st.set_page_config(
    page_title="ExperimentIQ | Decision Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1550px;
    }

    [data-testid="stSidebar"] {
        background: #101827;
    }

    [data-testid="stSidebar"] * {
        color: #edf2f7;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(115deg, #14213d, #254b70);
        color: white;
        margin-bottom: 20px;
    }

    .hero h1 {
        font-size: 2.35rem;
        margin: 0 0 5px 0;
        color: white;
    }

    .hero p {
        margin: 0;
        color: #d9e6f5;
        font-size: 1rem;
    }

    .eyebrow {
        font-size: .72rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: #93c5fd;
        font-weight: 700;
    }

    .section-note {
        color: #718096;
        font-size: .88rem;
        margin-top: -8px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 145, 170, .25);
        border-radius: 13px;
        padding: 13px 15px;
        background: rgba(128, 145, 170, .045);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(128, 145, 170, .2);
        border-radius: 10px;
    }

    div[data-testid="stPlotlyChart"] {
        border: 1px solid rgba(128, 145, 170, .15);
        border-radius: 12px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def show_chart(fig):
    """Display a Plotly chart using the current Streamlit width API."""
    if fig is not None:
        st.plotly_chart(fig, width="stretch")


def fmt_pct(value):
    """Format a percentage with an explicit sign."""
    try:
        number = float(value)
        if pd.isna(number):
            return "N/A"
        return f"{number:+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def safe_number(value, default=0):
    """Convert a value to a number without breaking the dashboard."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_metric_value(value, suffix=""):
    """Format a metric value safely."""
    number = safe_number(value, default=None)

    if number is None:
        return "N/A"

    return f"{number:,.2f}{suffix}"


def metric_help(name):
    explanations = {
        "Conversion rate": (
            "Share of assigned users with at least one valid "
            "conversion inside the 30-day outcome window."
        ),
        "Relative uplift": (
            "Percentage change in treatment conversion rate "
            "relative to the control conversion rate."
        ),
        "P-value": (
            "How compatible the observed difference is with a "
            "zero-effect hypothesis. It is not the probability "
            "that the hypothesis is true."
        ),
        "95% confidence interval": (
            "A range of plausible values for the absolute "
            "treatment-minus-control conversion difference."
        ),
    }

    return explanations.get(name, "")


def display_dataframe(data, **kwargs):
    """Consistent dataframe display across all dashboard pages."""

    if data is None:
        st.info("No records are available.")
        return

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    st.dataframe(
        data,
        width="stretch",
        hide_index=kwargs.pop("hide_index", True),
        **kwargs,
    )


def ground_truth_evaluation(frames, issues):
    """
    Compare detected issue categories against separate synthetic
    ground-truth labels.

    This is category-level coverage, not entity-level accuracy.
    """

    truth = frames.get("ground_truth", pd.DataFrame())

    if truth.empty:
        return pd.DataFrame(), 0, 0

    required_truth_columns = {
        "issue_type",
        "expected_decision",
    }

    if not required_truth_columns.issubset(truth.columns):
        return pd.DataFrame(), 0, 0

    detected = set()

    if (
        isinstance(issues, pd.DataFrame)
        and not issues.empty
        and "issue_type" in issues.columns
    ):
        detected = set(
            issues["issue_type"].dropna().astype(str).unique()
        )

    rows = []

    for _, item in truth.iterrows():
        issue_type = str(item["issue_type"])
        found = issue_type in detected

        rows.append({
            "Scenario": issue_type.replace("_", " ").title(),
            "Expected decision": str(
                item["expected_decision"]
            ).replace("_", " ").title(),
            "Detected": "Yes" if found else "No",
            "Status": "PASS" if found else "REVIEW",
        })

    result = pd.DataFrame(rows)

    return (
        result,
        int(result["Status"].eq("PASS").sum()),
        len(result),
    )


def analytical_assignments(frames, experiment_id):
    """
    Return assignment rows after applying eligibility and
    conflicting-variant exclusion rules.
    """

    assignments = frames["assignments"].copy()
    audience = frames["audience_members"]
    variants = frames["variants"]

    a = assignments[
        assignments["experiment_id"] == experiment_id
    ].copy()

    # Apply eligibility using the assignment-level field first.
    if "is_eligible" in a.columns:
        a = a[
            a["is_eligible"].fillna(False).astype(bool)
        ]

    elif "eligible" in a.columns:
        a = a[
            a["eligible"].fillna(False).astype(bool)
        ]

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

        a = a[
            a["eligible"].fillna(False).astype(bool)
        ]

    # Exclude users assigned to more than one variant.
    if not a.empty and {"user_id", "variant_id"}.issubset(a.columns):
        conflict_counts = (
            a.groupby("user_id")["variant_id"].nunique()
        )

        conflicting_users = conflict_counts[
            conflict_counts > 1
        ].index

        a = a[
            ~a["user_id"].isin(conflicting_users)
        ].copy()

    a = a.merge(
        variants[["variant_id", "variant_key", "name"]],
        on="variant_id",
        how="left",
        validate="many_to_one",
    )

    return a


def make_activity_chart(assigned, exposed, converted):
    """
    Compare activity counts without implying that all conversions
    must have a captured exposure event.
    """

    activity = pd.DataFrame({
        "Activity": [
            "Assigned",
            "Valid exposed",
            "Converted",
        ],
        "Unique assignments": [
            assigned,
            exposed,
            converted,
        ],
    })

    fig = px.bar(
        activity,
        x="Activity",
        y="Unique assignments",
        color="Activity",
        text="Unique assignments",
        title="Assignment and outcome activity",
        color_discrete_sequence=[
            "#315C8C",
            "#4C9B91",
            "#E3A64B",
        ],
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        height=370,
        margin=dict(l=15, r=15, t=55, b=25),
        showlegend=False,
        yaxis_title="Unique assignments",
        xaxis_title="",
    )

    return fig


def make_effect_plot(summaries):
    """Create a readable portfolio-wide horizontal uplift chart."""

    if summaries is None or summaries.empty:
        return None

    required = {"name", "uplift_pct"}

    if not required.issubset(summaries.columns):
        return None

    data = summaries.copy()

    data["uplift_pct"] = pd.to_numeric(
        data["uplift_pct"],
        errors="coerce",
    )

    data = data.dropna(subset=["uplift_pct"]).copy()

    if data.empty:
        return None

    decision_labels = {
        "PROMISING - REVIEW": "Promising",
        "INCONCLUSIVE": "Inconclusive",
        "DO NOT ROLL OUT - GUARDRAIL FAILURE": "Guardrail failure",
        "DO NOT ROLL OUT - SAMPLE RATIO MISMATCH": "SRM failure",
        "UNDERPERFORMS": "Underperforming",
    }

    if "decision" in data.columns:
        data["Decision"] = data["decision"].astype(str).map(
            lambda value: decision_labels.get(
                value.upper(),
                value.title(),
            )
        )
    else:
        data["Decision"] = "Not classified"

    colors = {
        "Promising": "#0875D1",
        "Inconclusive": "#77BDF2",
        "Guardrail failure": "#F28B82",
        "SRM failure": "#E53935",
        "Underperforming": "#8B5CF6",
        "Not classified": "#9CA3AF",
    }

    data = data.sort_values("uplift_pct", ascending=True)

    hover_columns = [
        column for column in ["p_value", "srm_status"]
        if column in data.columns
    ]

    fig = px.bar(
        data,
        x="uplift_pct",
        y="name",
        orientation="h",
        color="Decision",
        color_discrete_map=colors,
        hover_data=hover_columns,
        labels={
            "uplift_pct": "Relative uplift (%)",
            "name": "Experiment",
            "Decision": "Decision",
        },
        title="Treatment uplift across experiments",
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#64748B",
        line_width=1,
    )

    fig.update_layout(
        height=max(390, 48 * len(data) + 110),
        margin=dict(l=15, r=25, t=65, b=45),
        autosize=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        xaxis_title="Relative uplift (%)",
        yaxis_title="",
        legend_title_text="Decision",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(128,145,170,0.18)",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=False,
        automargin=True,
    )

    return fig


def make_decision_donut(summaries):
    """Create a compact portfolio decision distribution chart."""

    if (
        summaries is None
        or summaries.empty
        or "decision" not in summaries.columns
    ):
        return None

    labels = {
        "PROMISING - REVIEW": "Promising",
        "INCONCLUSIVE": "Inconclusive",
        "DO NOT ROLL OUT - GUARDRAIL FAILURE": "Guardrail failure",
        "DO NOT ROLL OUT - SAMPLE RATIO MISMATCH": "SRM failure",
        "UNDERPERFORMS": "Underperforming",
    }

    colors = {
        "Promising": "#0875D1",
        "Inconclusive": "#77BDF2",
        "Guardrail failure": "#F28B82",
        "SRM failure": "#E53935",
        "Underperforming": "#8B5CF6",
    }

    counts = (
        summaries["decision"]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis("Original decision")
        .reset_index(name="Experiments")
    )

    if counts.empty:
        return None

    counts["Decision"] = counts["Original decision"].map(
        lambda value: labels.get(
            value.upper(),
            value.title(),
        )
    )

    fig = px.pie(
        counts,
        names="Decision",
        values="Experiments",
        hole=0.62,
        color="Decision",
        color_discrete_map=colors,
        title="Portfolio decision distribution",
    )

    fig.update_traces(
        textposition="inside",
        texttemplate="%{percent:.0%}",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Experiments: %{value}<br>"
            "Share: %{percent}<extra></extra>"
        ),
    )

    fig.update_layout(
        height=390,
        margin=dict(l=15, r=15, t=65, b=75),
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    return fig


def prepare_derived_data(frames):
    """Run validation and analytics once per loaded dataset."""

    return {
        "issues": validate_frames(frames),
        "summaries": all_experiment_summaries(frames),
        "integrity": integrity_summary(frames),
    }


def safe_progress(numerator, denominator):
    """Return a valid progress value between zero and one."""
    if denominator <= 0:
        return 0.0

    return min(
        max(float(numerator) / float(denominator), 0.0),
        1.0,
    )


# =============================================================
# LOAD PERSISTED DATA
# =============================================================

if "frames" not in st.session_state:
    try:
        st.session_state.frames = (
            load_frames(DB_PATH)
            if DB_PATH.exists()
            else None
        )

    except Exception as exc:
        st.session_state.frames = None
        st.error(f"Could not load the saved dataset: {exc}")


if "derived" not in st.session_state:
    st.session_state.derived = None

if "export_zip" not in st.session_state:
    st.session_state.export_zip = None

if "reproducibility" not in st.session_state:
    st.session_state.reproducibility = None


# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:

    st.markdown("# 🧪 ExperimentIQ")
    st.caption("EXPERIMENTATION & DECISION INTELLIGENCE")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Experiment Manager",
            "Assignment Quality",
            "User Activity",
            "Results & Decisions",
            "Data Quality",
            "Data & Exports",
            "How it works",
        ],
    )

    st.divider()

    st.caption("DEMO DATASET")
    st.success("Synthetic data · No personal data")
    st.caption(f"Default seed · {SEED}")

    with st.expander("Dataset controls"):

        user_count = st.select_slider(
            "Audience size",
            options=[1000, 5000, 10000, 25000, 50000],
            value=50000,
        )

        seed = st.number_input(
            "Random seed",
            value=int(SEED),
            step=1,
        )

        if st.button(
            "Generate dataset",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner(
                    "Building linked experiment records..."
                ):
                    new_frames = generate_dataset(
                        int(user_count),
                        10,
                        int(seed),
                    )

                    save_frames(new_frames, DB_PATH)

                    st.session_state.frames = load_frames(DB_PATH)
                    st.session_state.derived = None
                    st.session_state.export_zip = None
                    st.session_state.reproducibility = None

                st.success("Dataset generated and saved.")
                st.rerun()

            except Exception as exc:
                st.error(f"Dataset generation failed: {exc}")

        st.caption(
            "Generating a new dataset replaces source tables. "
            "Versioned experiment-result snapshots are retained. "
            "Export source data before generating a replacement "
            "if you need to preserve an older dataset."
        )


# =============================================================
# HERO
# =============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            Experimentation & Decision Intelligence
        </div>
        <h1>ExperimentIQ</h1>
        <p>
            Design experiments, validate data quality, understand
            treatment effects and make evidence-led decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# EMPTY WORKSPACE
# =============================================================

frames = st.session_state.frames

if frames is None:

    st.info(
        "Your workspace is ready. Generate the synthetic dataset "
        "to explore the experimentation workflow."
    )

    a, b, c = st.columns(3)

    a.metric("Target audience", "50,000")
    b.metric("Experiments", "10")
    c.metric("Reproducibility seed", str(SEED))

    st.markdown("### What this application helps you do")

    x, y, z = st.columns(3)

    x.markdown(
        "**1 · Design**\n\n"
        "Define a hypothesis, primary metric, audience and allocation."
    )

    y.markdown(
        "**2 · Validate**\n\n"
        "Check assignment quality, event order, telemetry and guardrails."
    )

    z.markdown(
        "**3 · Decide**\n\n"
        "Estimate treatment effect, uncertainty and rollout risks."
    )

    if st.button(
        "Create the demo dataset",
        type="primary",
        width="stretch",
    ):
        try:
            with st.spinner(
                "Generating linked audience and event records..."
            ):
                generated = generate_dataset(
                    50000,
                    10,
                    int(SEED),
                )

                save_frames(generated, DB_PATH)

                st.session_state.frames = load_frames(DB_PATH)
                st.session_state.derived = None

            st.rerun()

        except Exception as exc:
            st.error(f"Could not create the demo dataset: {exc}")

    st.stop()


# =============================================================
# DATA PREPARATION AND CACHING
# =============================================================

required_tables = [
    "experiments",
    "variants",
    "audience_members",
    "assignments",
    "exposure_events",
    "conversion_events",
    "metric_definitions",
    "guardrail_metrics",
    "guardrail_observations",
    "experiment_results",
    "ground_truth",
]

missing_tables = [
    name for name in required_tables
    if name not in frames
]

if missing_tables:
    st.error(
        "The saved dataset is missing required tables: "
        + ", ".join(missing_tables)
    )
    st.stop()


experiments = frames["experiments"]
assignments = frames["assignments"]
conversions = frames["conversion_events"]
exposures = frames["exposure_events"]
variants = frames["variants"]

if experiments.empty:
    st.error(
        "The experiment registry is empty. Generate the dataset again."
    )
    st.stop()


if st.session_state.derived is None:
    try:
        with st.spinner(
            "Preparing dashboard summaries and data-quality checks..."
        ):
            st.session_state.derived = prepare_derived_data(frames)

    except Exception as exc:
        st.error(
            f"Could not prepare dashboard analytics: {exc}"
        )
        st.stop()


issues = st.session_state.derived["issues"]
summaries = st.session_state.derived["summaries"]
integrity = st.session_state.derived["integrity"]


truth_eval, truth_pass, truth_total = ground_truth_evaluation(
    frames,
    issues,
)


exp_lookup = experiments.set_index("experiment_id")

selected = st.selectbox(
    "Choose an experiment to inspect",
    experiments["experiment_id"].tolist(),
    format_func=lambda value: (
        f"{value} · {exp_lookup.loc[value, 'name']}"
    ),
)

selected_exp = exp_lookup.loc[selected]


# =============================================================
# SELECTED EXPERIMENT ANALYSIS
# =============================================================

try:
    result = analyze_experiment(
        selected,
        assignments,
        conversions,
        exposures,
        variants,
        frames["guardrail_metrics"],
        experiments,
        guardrail_observations=frames.get(
            "guardrail_observations",
            pd.DataFrame(),
        ),
        metric_definitions=frames.get(
            "metric_definitions",
            pd.DataFrame(),
        ),
    )

    clean_assignments = analytical_assignments(
        frames,
        selected,
    )

except Exception as exc:
    st.error(
        f"Could not analyze the selected experiment: {exc}"
    )
    st.stop()


# =============================================================
# 1. DASHBOARD
# =============================================================

if page == "Dashboard":

    st.subheader("Portfolio overview")

    st.caption(
        "A portfolio-level view of experiment performance, "
        "evidence quality and exceptions."
    )

    k1, k2, k3, k4 = st.columns(4)

    audience = frames["audience_members"]

    if "eligible" in audience.columns:
        eligible_count = int(
            audience["eligible"].fillna(False).astype(bool).sum()
        )
    else:
        eligible_count = 0

    k1.metric("Experiments", f"{len(experiments):,}")

    k2.metric("Eligible audience", f"{eligible_count:,}")

    k3.metric(
        "Assignments",
        f"{len(assignments):,}",
        help="Includes deliberately labelled negative scenarios.",
    )

    k4.metric(
        "Data-quality findings",
        f"{len(issues):,}",
        help="Detected data-quality exceptions.",
    )

    st.divider()

    st.markdown("### Experiment performance")

    left, right = st.columns([1.6, 1])

    with left:
        if (
            not summaries.empty
            and {"name", "uplift_pct"}.issubset(summaries.columns)
        ):
            chart_data = summaries.copy()

            chart_data["uplift_pct"] = pd.to_numeric(
                chart_data["uplift_pct"],
                errors="coerce",
            )

            chart_data = chart_data.dropna(
                subset=["uplift_pct"]
            )

            if not chart_data.empty:
                hover_columns = [
                    column for column in ["p_value", "srm_status"]
                    if column in chart_data.columns
                ]

                fig = px.bar(
                    chart_data,
                    x="name",
                    y="uplift_pct",
                    color="decision" if "decision" in chart_data.columns else None,
                    hover_data=hover_columns,
                    labels={
                        "name": "Experiment",
                        "uplift_pct": "Relative uplift (%)",
                        "decision": "Decision",
                    },
                    title="Relative treatment uplift",
                )

                fig.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color="gray",
                )

                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=55, b=90),
                    xaxis_tickangle=-25,
                    legend_title_text="Decision",
                )

                show_chart(fig)
            else:
                st.info("No numeric uplift values are available.")
        else:
            st.info("No experiment performance results are available.")

    with right:
        decision_fig = make_decision_donut(summaries)

        if decision_fig is not None:
            show_chart(decision_fig)
        else:
            st.info("No decision records are available.")

    st.markdown("### Treatment-effect comparison")

    effect_fig = make_effect_plot(summaries)

    if effect_fig is not None:
        show_chart(effect_fig)
    else:
        st.info("No treatment-effect comparison is available.")

    st.markdown("### Evidence and quality")

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Known scenarios detected",
        f"{truth_pass}/{truth_total}",
    )

    high_count = (
        int(issues["severity"].eq("HIGH").sum())
        if not issues.empty and "severity" in issues.columns
        else 0
    )

    q2.metric(
        "High-severity findings",
        f"{high_count:,}",
    )

    passed_checks = (
        int(integrity["status"].eq("PASS").sum())
        if not integrity.empty and "status" in integrity.columns
        else 0
    )

    q3.metric(
        "Database checks passed",
        f"{passed_checks}/{len(integrity)}",
    )

    st.progress(
        safe_progress(truth_pass, truth_total)
    )

    st.caption(
        "Ground-truth labels are stored separately from analytical inputs. "
        "Scenario detection coverage is evaluated by issue category, "
        "not by matching every individual anomalous record."
    )

    st.markdown("### Experiments at a glance")

    view = summaries.rename(columns={
        "name": "Experiment",
        "assigned": "Assigned users",
        "uplift_pct": "Relative uplift (%)",
        "p_value": "P-value",
        "srm_status": "Randomization",
        "decision": "Decision",
    })

    display_dataframe(view)

    st.info(
        "**Interpretation:** Positive uplift means treatment converted "
        "better than control. It is not sufficient by itself to recommend "
        "rollout. Check uncertainty, randomization and guardrails."
    )


# =============================================================
# 2. EXPERIMENT MANAGER
# =============================================================

elif page == "Experiment Manager":

    st.subheader("Experiment manager")

    st.caption(
        "Review the experiment registry, hypotheses, variants "
        "and predeclared metric definitions."
    )

    registry = experiments.rename(columns={
        "experiment_id": "Experiment ID",
        "name": "Name",
        "hypothesis": "Hypothesis",
        "primary_metric": "Primary metric",
        "status": "Lifecycle status",
        "allocation_a": "Control allocation",
        "allocation_b": "Treatment allocation",
    })

    display_dataframe(registry)

    st.markdown("### Selected experiment design")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Lifecycle",
        str(selected_exp.get("status", "Unknown")).title(),
    )

    c2.metric(
        "Primary metric",
        str(selected_exp.get("primary_metric", "Unknown"))
        .replace("_", " ")
        .title(),
    )

    c3.metric(
        "Planned split",
        f"{safe_number(selected_exp.get('allocation_a')):.0%} / "
        f"{safe_number(selected_exp.get('allocation_b')):.0%}",
    )

    st.markdown("**Hypothesis**")
    st.info(str(selected_exp.get("hypothesis", "No hypothesis recorded.")))

    st.markdown("**Variants**")

    selected_variants = variants[
        variants["experiment_id"] == selected
    ]

    display_dataframe(selected_variants)

    st.markdown("**Metric definition history**")

    metric_history = frames["metric_definitions"][
        frames["metric_definitions"]["experiment_id"] == selected
    ]

    display_dataframe(metric_history)

    st.caption(
        "The current prototype reads experiment definitions from "
        "generated seed data. Interactive experiment authoring "
        "is a future extension."
    )


# =============================================================
# 3. ASSIGNMENT QUALITY
# =============================================================

elif page == "Assignment Quality":

    st.subheader("Assignment quality")

    st.caption(
        "Compare observed allocation against the experiment's "
        "predeclared allocation plan."
    )

    raw = assignments[
        assignments["experiment_id"] == selected
    ].merge(
        variants[["variant_id", "variant_key", "name"]],
        on="variant_id",
        how="left",
    )

    counts = (
        clean_assignments["variant_key"]
        .value_counts()
        .reindex(["A", "B"], fill_value=0)
    )

    raw_conflicts = (
        raw.groupby("user_id")["variant_id"].nunique().gt(1).sum()
        if not raw.empty
        else 0
    )

    ca, cb, cc, cd = st.columns(4)

    ca.metric("Raw assignments", f"{len(raw):,}")
    cb.metric("Valid control", f"{counts['A']:,}")
    cc.metric("Valid treatment", f"{counts['B']:,}")
    cd.metric("Conflicting users", f"{raw_conflicts:,}")

    st.markdown("### Observed versus expected allocation")

    expected_a = safe_number(
        result.get("expected_allocation_a", 0.5),
        0.5,
    )

    expected_b = safe_number(
        result.get("expected_allocation_b", 0.5),
        0.5,
    )

    chart = pd.DataFrame({
        "Variant": ["Control", "Treatment"],
        "Observed users": [
            counts["A"],
            counts["B"],
        ],
        "Expected users": [
            len(clean_assignments) * expected_a,
            len(clean_assignments) * expected_b,
        ],
    })

    fig = px.bar(
        chart,
        x="Variant",
        y=["Observed users", "Expected users"],
        barmode="group",
        title="Valid observed versus expected assignment",
        color_discrete_sequence=["#315C8C", "#E3A64B"],
    )

    fig.update_layout(height=390)

    show_chart(fig)

    st.markdown("### Allocation proportions")

    denominator = max(len(clean_assignments), 1)

    observed = pd.DataFrame({
        "Variant": ["Control", "Treatment"],
        "Observed share": [
            counts["A"] / denominator,
            counts["B"] / denominator,
        ],
        "Configured share": [
            expected_a,
            expected_b,
        ],
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Observed",
        x=observed["Variant"],
        y=observed["Observed share"] * 100,
    ))

    fig.add_trace(go.Bar(
        name="Configured",
        x=observed["Variant"],
        y=observed["Configured share"] * 100,
    ))

    fig.update_layout(
        barmode="group",
        title="Allocation share comparison",
        yaxis_title="Share (%)",
        height=350,
    )

    show_chart(fig)

    x, y = st.columns(2)

    x.metric(
        "SRM p-value",
        safe_metric_value(result.get("srm_p_value")),
    )

    y.metric(
        "SRM status",
        str(result.get("srm_status", "N/A")),
    )

    if result.get("srm_status") == "FAIL":
        st.error(
            "Sample-ratio mismatch detected. Investigate allocation "
            "before trusting the treatment estimate."
        )
    else:
        st.success(
            "No sample-ratio mismatch detected at the configured threshold."
        )

    st.markdown("### Assignment exception records")

    subset = (
        issues[
            issues["issue_type"].isin([
                "MULTIPLE_VARIANT_ASSIGNMENT",
                "SAMPLE_RATIO_MISMATCH",
            ])
        ]
        if not issues.empty and "issue_type" in issues.columns
        else pd.DataFrame()
    )

    display_dataframe(subset)

    with st.expander("View raw assignment records"):
        display_dataframe(raw.head(500))


# =============================================================
# 4. USER ACTIVITY
# =============================================================

elif page == "User Activity":

    st.subheader("User activity & event tracking")

    st.caption(
        "Inspect assignment, exposure telemetry and conversion events "
        "using the 30-day outcome window."
    )

    # ---------------------------------------------------------
    # Valid assignments
    # ---------------------------------------------------------

    a = clean_assignments[
        ["assignment_id", "user_id", "assigned_at", "variant_key"]
    ].copy()

    a["assignment_id"] = a["assignment_id"].astype(str)

    a["assigned_at"] = pd.to_datetime(
        a["assigned_at"],
        utc=True,
        errors="coerce",
    )

    valid_ids = set(a["assignment_id"])

    # ---------------------------------------------------------
    # Exposure events
    # ---------------------------------------------------------

    e = exposures[
        exposures["experiment_id"] == selected
    ].copy()

    e["assignment_id"] = e["assignment_id"].astype(str)

    e = e[
        e["assignment_id"].isin(valid_ids)
    ].merge(
        a[["assignment_id", "assigned_at"]],
        on="assignment_id",
        how="inner",
        validate="many_to_one",
    )

    e["exposed_at"] = pd.to_datetime(
        e["exposed_at"],
        utc=True,
        errors="coerce",
    )

    missing_telemetry = e[
        e["telemetry_status"].astype(str).str.lower().eq("missing")
    ].copy()

    captured = e[
        e["telemetry_status"].astype(str).str.lower().eq("captured")
    ].copy()

    valid_exposures = captured[
        captured["exposed_at"].notna()
        & captured["assigned_at"].notna()
        & (captured["exposed_at"] >= captured["assigned_at"])
    ].copy()

    invalid_exposure_events = captured[
        captured["exposed_at"].isna()
        | captured["assigned_at"].isna()
        | (captured["exposed_at"] < captured["assigned_at"])
    ].copy()

    # ---------------------------------------------------------
    # Conversion events
    # ---------------------------------------------------------

    c = conversions[
        conversions["experiment_id"] == selected
    ].copy()

    c["assignment_id"] = c["assignment_id"].astype(str)

    c = c[
        c["assignment_id"].isin(valid_ids)
    ].merge(
        a[["assignment_id", "assigned_at"]],
        on="assignment_id",
        how="inner",
        validate="many_to_one",
    )

    c["converted_at"] = pd.to_datetime(
        c["converted_at"],
        utc=True,
        errors="coerce",
    )

    c["window_end"] = (
        c["assigned_at"] + pd.Timedelta(days=30)
    )

    valid_timestamp = (
        c["converted_at"].notna()
        & c["assigned_at"].notna()
    )

    after_assignment = (
        valid_timestamp
        & (c["converted_at"] >= c["assigned_at"])
    )

    within_window = (
        after_assignment
        & (c["converted_at"] <= c["window_end"])
    )

    valid_conversions = c[
        within_window
    ].copy()

    # Count each assignment once in the activity summary.
    valid_conversions = (
        valid_conversions.sort_values("converted_at")
        .drop_duplicates(
            subset=["assignment_id"],
            keep="first",
        )
    )

    invalid_timestamp_conversions = c[
        ~valid_timestamp
    ].copy()

    before_assignment_conversions = c[
        valid_timestamp
        & (c["converted_at"] < c["assigned_at"])
    ].copy()

    outside_window_conversions = c[
        after_assignment
        & (c["converted_at"] > c["window_end"])
    ].copy()

    # ---------------------------------------------------------
    # Activity KPIs
    # ---------------------------------------------------------

    assigned_count = a["assignment_id"].nunique()

    exposed_count = (
        valid_exposures["assignment_id"].nunique()
    )

    converted_count = (
        valid_conversions["assignment_id"].nunique()
    )

    missing_count = (
        missing_telemetry["assignment_id"].nunique()
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Assigned", f"{assigned_count:,}")
    m2.metric("Valid exposed", f"{exposed_count:,}")
    m3.metric("Converted assignments", f"{converted_count:,}")
    m4.metric("Missing telemetry", f"{missing_count:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        show_chart(
            make_activity_chart(
                assigned_count,
                exposed_count,
                converted_count,
            )
        )

    with right:
        telemetry_summary = pd.DataFrame({
            "Telemetry status": ["Captured", "Missing"],
            "Event records": [
                len(captured),
                len(missing_telemetry),
            ],
        })

        fig = px.pie(
            telemetry_summary,
            names="Telemetry status",
            values="Event records",
            hole=0.55,
            color="Telemetry status",
            color_discrete_map={
                "Captured": "#4C9B91",
                "Missing": "#E3A64B",
            },
            title="Exposure telemetry completeness",
        )

        fig.update_layout(height=370)

        show_chart(fig)

    st.markdown("### Event-quality checks")

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Valid captured exposures",
        f"{len(valid_exposures):,}",
    )

    q2.metric(
        "Outside 30-day window",
        f"{len(outside_window_conversions):,}",
    )

    q3.metric(
        "Before assignment",
        f"{len(before_assignment_conversions):,}",
    )

    q4.metric(
        "Invalid timestamps",
        f"{len(invalid_timestamp_conversions):,}",
    )

    st.caption(
        "The activity chart compares unique assignment counts; the stages "
        "are not necessarily nested. Exposure requires captured telemetry "
        "and an event timestamp at or after assignment. Conversion requires "
        "an event within 30 days of assignment. The primary treatment "
        "analysis remains intention-to-treat and does not require a captured "
        "exposure event."
    )

    with st.expander("Valid captured exposure events"):
        display_dataframe(
            valid_exposures.drop(
                columns=["assigned_at"],
                errors="ignore",
            ).head(250)
        )

    with st.expander("Missing telemetry records"):
        display_dataframe(
            missing_telemetry.drop(
                columns=["assigned_at"],
                errors="ignore",
            ).head(250)
        )

    with st.expander("Invalid exposure records"):
        display_dataframe(
            invalid_exposure_events.drop(
                columns=["assigned_at"],
                errors="ignore",
            ).head(250)
        )

    with st.expander("Valid conversion assignments"):
        display_dataframe(
            valid_conversions.drop(
                columns=["assigned_at", "window_end"],
                errors="ignore",
            ).head(250)
        )

    with st.expander("Conversions outside the 30-day window"):
        display_dataframe(
            outside_window_conversions.drop(
                columns=["assigned_at", "window_end"],
                errors="ignore",
            ).head(250)
        )

    with st.expander("Conversions before assignment"):
        display_dataframe(
            before_assignment_conversions.drop(
                columns=["assigned_at", "window_end"],
                errors="ignore",
            ).head(250)
        )

    with st.expander("Conversions with invalid timestamps"):
        display_dataframe(
            invalid_timestamp_conversions.drop(
                columns=["assigned_at", "window_end"],
                errors="ignore",
            ).head(250)
        )


# =============================================================
# 5. RESULTS AND DECISIONS
# =============================================================

elif page == "Results & Decisions":

    st.subheader("Results & decisions")

    st.caption(
        "Treatment effects are estimated by original assignment "
        "(intention-to-treat), not by exposure compliance."
    )

    variant_results = result.get(
        "variants",
        pd.DataFrame(),
    ).copy()

    if variant_results.empty or "variant" not in variant_results.columns:
        st.error("Variant-level analysis results are unavailable.")
        st.stop()

    v = variant_results.set_index("variant")

    if not {"A", "B"}.issubset(set(v.index)):
        st.error(
            "The selected experiment does not contain both "
            "control and treatment analysis results."
        )
        st.stop()

    ca, cb, cc = st.columns(3)

    ca.metric(
        "Control conversion",
        f"{safe_number(v.loc['A'].get('conversion_rate')):.2%}",
    )

    cb.metric(
        "Treatment conversion",
        f"{safe_number(v.loc['B'].get('conversion_rate')):.2%}",
    )

    cc.metric(
        "Relative uplift",
        fmt_pct(result.get("uplift_pct")),
        help=metric_help("Relative uplift"),
    )

    st.markdown("### Conversion rate by variant")

    rates = variant_results.copy()

    required_rate_columns = {
        "conversion_rate",
        "ci_low",
        "ci_high",
    }

    if required_rate_columns.issubset(rates.columns):

        rates["Conversion rate (%)"] = (
            pd.to_numeric(
                rates["conversion_rate"],
                errors="coerce",
            ) * 100
        )

        rates["Upper error"] = (
            pd.to_numeric(rates["ci_high"], errors="coerce")
            - pd.to_numeric(rates["conversion_rate"], errors="coerce")
        ).clip(lower=0) * 100

        rates["Lower error"] = (
            pd.to_numeric(rates["conversion_rate"], errors="coerce")
            - pd.to_numeric(rates["ci_low"], errors="coerce")
        ).clip(lower=0) * 100

        name_column = (
            "name" if "name" in rates.columns else "variant"
        )

        fig = px.bar(
            rates,
            x=name_column,
            y="Conversion rate (%)",
            color=name_column,
            title="Conversion rate with 95% Wilson intervals",
            color_discrete_sequence=["#315C8C", "#4C9B91"],
        )

        fig.update_traces(
            error_y=dict(
                type="data",
                array=rates["Upper error"],
                arrayminus=rates["Lower error"],
                visible=True,
            )
        )

        fig.update_layout(
            height=390,
            showlegend=False,
            xaxis_title="Variant",
        )

        show_chart(fig)

    else:
        st.warning(
            "Confidence interval columns are unavailable for the "
            "variant conversion chart."
        )

    st.markdown("### Treatment effect and uncertainty")

    x, y, z = st.columns(3)

    x.metric(
        "P-value",
        safe_metric_value(result.get("p_value")),
        help=metric_help("P-value"),
    )

    ci_low = safe_number(result.get("ci_low"), default=None)
    ci_high = safe_number(result.get("ci_high"), default=None)

    if ci_low is not None and ci_high is not None:
        ci_display = f"[{ci_low:.2%}, {ci_high:.2%}]"
    else:
        ci_display = "N/A"

    y.metric(
        "95% CI · absolute difference",
        ci_display,
        help=metric_help("95% confidence interval"),
    )

    z.metric(
        "Randomization",
        str(result.get("srm_status", "N/A")),
    )

    st.markdown("### Decision recommendation")

    decision = str(
        result.get("decision", "INCONCLUSIVE")
    ).upper()

    st.markdown(
        f"## {result.get('decision', 'INCONCLUSIVE')}"
    )

    if "DO NOT ROLL OUT" in decision:
        st.error(
            "A randomization or guardrail issue was detected. "
            "Do not use this result for a rollout decision until investigated."
        )

    elif "PROMISING" in decision:
        st.success(
            "The observed treatment effect is positive and statistically "
            "significant. Review guardrails and operational context before rollout."
        )

    elif "UNDERPERFORMS" in decision:
        st.warning(
            "The treatment is statistically worse than control "
            "in this synthetic analysis."
        )

    else:
        st.warning(
            "The evidence is inconclusive. The observed difference "
            "may be due to sampling variation."
        )

    st.markdown("### Guardrail checks")

    guardrails = result.get(
        "guardrails",
        pd.DataFrame(),
    )

    if isinstance(guardrails, pd.DataFrame) and not guardrails.empty:

        display_dataframe(
            guardrails.rename(columns={
                "metric": "Guardrail",
                "observed": "Observed value",
                "threshold": "Allowed threshold",
                "status": "Status",
            })
        )

        observations = frames.get(
            "guardrail_observations",
            pd.DataFrame(),
        )

        if (
            not observations.empty
            and "assignment_id" in observations.columns
        ):
            assigned_for_guardrails = clean_assignments[
                ["assignment_id", "variant_key"]
            ].drop_duplicates("assignment_id")

            go_data = observations.merge(
                assigned_for_guardrails,
                on="assignment_id",
                how="inner",
                validate="many_to_one",
            )

            metric_fields = [
                field for field in [
                    "error_occurred",
                    "refund_occurred",
                ]
                if field in go_data.columns
            ]

            if metric_fields:
                plot_rows = []

                for field in metric_fields:
                    rates_by_variant = (
                        go_data.groupby("variant_key")[field]
                        .mean()
                        .mul(100)
                    )

                    for variant_key, rate in rates_by_variant.items():
                        plot_rows.append({
                            "Guardrail": field.replace(
                                "_", " "
                            ).title(),
                            "Variant": (
                                "Control" if variant_key == "A"
                                else "Treatment"
                            ),
                            "Observed rate (%)": rate,
                        })

                if plot_rows:
                    guardrail_plot = pd.DataFrame(plot_rows)

                    fig = px.bar(
                        guardrail_plot,
                        x="Guardrail",
                        y="Observed rate (%)",
                        color="Variant",
                        barmode="group",
                        title="Observed guardrail rates by variant",
                        color_discrete_sequence=["#315C8C", "#E3A64B"],
                    )

                    fig.update_layout(height=370)

                    show_chart(fig)

    else:
        st.info(
            "No guardrail results are available for this experiment."
        )

    st.caption(
        "Guardrail indicators use generated synthetic observations. "
        "Statistical estimates are intended for demonstration, not "
        "production causal inference."
    )

    st.markdown("### Stored result snapshots")

    snapshots = frames["experiment_results"][
        frames["experiment_results"]["experiment_id"] == selected
    ]

    display_dataframe(snapshots)


# =============================================================
# 6. DATA QUALITY
# =============================================================

elif page == "Data Quality":

    st.subheader("Data quality & exception queue")

    st.caption(
        "Review detected exceptions, severity and known-scenario coverage."
    )

    if issues.empty:
        high = medium = low = 0
    else:
        high = (
            int(issues["severity"].eq("HIGH").sum())
            if "severity" in issues.columns else 0
        )

        medium = (
            int(issues["severity"].eq("MEDIUM").sum())
            if "severity" in issues.columns else 0
        )

        low = (
            int(issues["severity"].eq("LOW").sum())
            if "severity" in issues.columns else 0
        )

    x, y, z, w = st.columns(4)

    x.metric("Total findings", f"{len(issues):,}")
    y.metric("High severity", f"{high:,}")
    z.metric("Medium severity", f"{medium:,}")
    w.metric("Low severity", f"{low:,}")

    if not issues.empty:

        left, right = st.columns(2)

        with left:
            if "severity" in issues.columns:
                sev = (
                    issues["severity"]
                    .value_counts()
                    .rename_axis("Severity")
                    .reset_index(name="Findings")
                )

                fig = px.bar(
                    sev,
                    x="Severity",
                    y="Findings",
                    color="Severity",
                    title="Findings by severity",
                    color_discrete_map={
                        "HIGH": "#D9534F",
                        "MEDIUM": "#E3A64B",
                        "LOW": "#4C9B91",
                    },
                )

                fig.update_layout(height=350)

                show_chart(fig)

        with right:
            if "issue_type" in issues.columns:
                issue_counts = (
                    issues["issue_type"]
                    .value_counts()
                    .rename_axis("Issue type")
                    .reset_index(name="Findings")
                    .sort_values("Findings")
                )

                fig = px.bar(
                    issue_counts,
                    x="Findings",
                    y="Issue type",
                    orientation="h",
                    title="Most frequent data-quality issues",
                )

                fig.update_layout(height=350)

                show_chart(fig)

        if {"severity", "issue_type"}.issubset(issues.columns):

            f1, f2 = st.columns(2)

            with f1:
                sev_filter = st.multiselect(
                    "Severity",
                    ["HIGH", "MEDIUM", "LOW"],
                    default=["HIGH", "MEDIUM", "LOW"],
                )

            with f2:
                type_options = sorted(
                    issues["issue_type"].dropna().unique()
                )

                type_filter = st.multiselect(
                    "Issue type",
                    type_options,
                    default=type_options,
                )

            filtered = issues[
                issues["severity"].isin(sev_filter)
                & issues["issue_type"].isin(type_filter)
            ]

            display_dataframe(filtered)

        else:
            display_dataframe(issues)

    else:
        st.success("No data-quality findings detected.")

    st.markdown("### Known-scenario evaluation")

    st.metric(
        "Ground-truth detection coverage",
        f"{truth_pass}/{truth_total}",
    )

    if truth_total:
        st.progress(
            safe_progress(truth_pass, truth_total)
        )

    display_dataframe(truth_eval)

    st.caption(
        "Ground-truth labels are not included in analytical inputs. "
        "Detection coverage is category-level in this demonstration."
    )

    with st.expander("Database integrity checks"):
        display_dataframe(integrity)


# =============================================================
# 7. DATA AND EXPORTS
# =============================================================

elif page == "Data & Exports":

    st.subheader("Data, traceability & reproducibility")

    st.caption(
        "Inspect generated tables, export source records and "
        "reproduce the same dataset using the same seed."
    )

    table_names = [
        "experiments",
        "variants",
        "audience_members",
        "assignments",
        "exposure_events",
        "conversion_events",
        "metric_definitions",
        "guardrail_metrics",
        "guardrail_observations",
        "experiment_results",
        "ground_truth",
    ]

    inv = pd.DataFrame([
        {
            "Table": table,
            "Rows": len(frames[table]),
            "Columns": len(frames[table].columns),
        }
        for table in table_names
    ])

    display_dataframe(inv)

    st.markdown("### Preview a source table")

    chosen = st.selectbox(
        "Table",
        table_names,
    )

    display_dataframe(
        frames[chosen].head(100)
    )

    st.markdown("### Export complete dataset")

    if st.button(
        "Prepare CSV + JSON export bundle",
        type="primary",
        width="stretch",
    ):
        try:
            with st.spinner(
                "Preparing export bundle and SHA-256 manifest..."
            ):
                with tempfile.TemporaryDirectory() as tmp:
                    export_bundle(frames, tmp)

                    buf = io.BytesIO()

                    with zipfile.ZipFile(
                        buf,
                        "w",
                        zipfile.ZIP_DEFLATED,
                    ) as zf:
                        for path in Path(tmp).rglob("*"):
                            if path.is_file():
                                zf.write(
                                    path,
                                    path.relative_to(tmp),
                                )

                    st.session_state.export_zip = buf.getvalue()

            st.success("Export bundle prepared successfully.")

        except Exception as exc:
            st.error(f"Export failed: {exc}")

    if st.session_state.get("export_zip"):

        st.download_button(
            "Download export bundle (.zip)",
            data=st.session_state.export_zip,
            file_name="ExperimentIQ_exports.zip",
            mime="application/zip",
            width="stretch",
        )

    st.warning(
        "Ground truth is exported as its own file. "
        "Do not merge it into model or analysis input features."
    )

    st.markdown("### Reproducibility settings")

    dataset_seed = (
        int(experiments["seed"].iloc[0])
        if "seed" in experiments.columns and not experiments.empty
        else int(SEED)
    )

    st.code(
        f"Seed: {dataset_seed}\n"
        f"Experiments: {len(experiments)}\n"
        f"Audience rows: {len(frames['audience_members']):,}",
        language="text",
    )

    if st.button(
        "Verify fixed-seed reproducibility",
        width="stretch",
    ):
        try:
            with st.spinner(
                "Generating the same dataset twice and comparing "
                "SHA-256 checksums..."
            ):
                dataset_size = len(
                    frames["audience_members"]
                )

                experiment_count = len(experiments)

                run_a = generate_dataset(
                    dataset_size,
                    experiment_count,
                    dataset_seed,
                )

                run_b = generate_dataset(
                    dataset_size,
                    experiment_count,
                    dataset_seed,
                )

                reproducibility_rows = []

                for table in run_a:

                    if table not in run_b:
                        reproducibility_rows.append({
                            "Table": table,
                            "Rows in run A": len(run_a[table]),
                            "Rows in run B": "Missing",
                            "Checksum match": False,
                        })
                        continue

                    hash_a = hashlib.sha256(
                        run_a[table].to_csv(
                            index=False
                        ).encode("utf-8")
                    ).hexdigest()

                    hash_b = hashlib.sha256(
                        run_b[table].to_csv(
                            index=False
                        ).encode("utf-8")
                    ).hexdigest()

                    reproducibility_rows.append({
                        "Table": table,
                        "Rows in run A": len(run_a[table]),
                        "Rows in run B": len(run_b[table]),
                        "Checksum match": hash_a == hash_b,
                    })

                st.session_state.reproducibility = pd.DataFrame(
                    reproducibility_rows
                )

        except Exception as exc:
            st.error(
                f"Reproducibility check failed: {exc}"
            )

    if st.session_state.get("reproducibility") is not None:

        rep = st.session_state.reproducibility

        display_dataframe(rep)

        if (
            not rep.empty
            and rep["Checksum match"].fillna(False).all()
        ):
            st.success(
                "Reproducibility verified: all generated tables match."
            )
        else:
            st.error(
                "A checksum mismatch was found. Review affected tables."
            )

    st.markdown("### Integrity summary")

    display_dataframe(integrity)


# =============================================================
# 8. HOW IT WORKS
# =============================================================

elif page == "How it works":

    st.subheader("How ExperimentIQ works")

    st.caption(
        "Understand the business problem, users, relationships "
        "and end-to-end workflow."
    )

    st.markdown("### Business problem")

    st.write(
        "Teams run experiments to learn whether a new experience "
        "performs better than the current one. A trustworthy decision "
        "requires correct assignment, valid event data, healthy "
        "operational guardrails and visible statistical uncertainty."
    )

    st.markdown("### Who uses the system?")

    actor_cols = st.columns(5)

    actors = [
        (
            "Customer / prospect",
            "Receives a control or treatment experience.",
        ),
        (
            "Marketing / sales ops",
            "Defines and launches an experiment.",
        ),
        (
            "Support agent",
            "Helps resolve user or experience issues.",
        ),
        (
            "Analyst",
            "Checks evidence and interprets results.",
        ),
        (
            "Administrator",
            "Maintains data and system configuration.",
        ),
    ]

    for col, (name, desc) in zip(actor_cols, actors):
        with col:
            st.markdown(f"**{name}**")
            st.caption(desc)

    st.markdown("### End-to-end workflow")

    steps = [
        (
            "1 · Define",
            "Record the hypothesis, primary metric, guardrails, "
            "audience and allocation plan.",
        ),
        (
            "2 · Assign",
            "Select eligible audience members and randomly allocate "
            "them using a fixed seed.",
        ),
        (
            "3 · Track",
            "Record exposures and conversions using valid assignment IDs.",
        ),
        (
            "4 · Validate",
            "Check keys, relationships, event order, outcome windows, "
            "sample ratios and metric versions.",
        ),
        (
            "5 · Analyze",
            "Calculate conversion rates, treatment uplift, confidence "
            "intervals, p-values and guardrail status.",
        ),
        (
            "6 · Decide",
            "Show recommendations with exceptions and traceable "
            "result snapshots.",
        ),
    ]

    for title, desc in steps:
        st.markdown(f"**{title}**")
        st.write(desc)

    st.markdown("### Core data relationships")

    relationships = pd.DataFrame([
        [
            "Experiment",
            "Variant",
            "1 : many",
            "Each experiment has control and treatment variants.",
        ],
        [
            "Experiment",
            "Assignment",
            "1 : many",
            "An experiment has many user assignments.",
        ],
        [
            "AudienceMember",
            "Assignment",
            "1 : many over time",
            "A user may participate in multiple experiments.",
        ],
        [
            "Variant",
            "Assignment",
            "1 : many",
            "Each assignment references one variant.",
        ],
        [
            "Assignment",
            "ExposureEvent",
            "1 : many",
            "An assignment can have zero or more exposure events.",
        ],
        [
            "Assignment",
            "ConversionEvent",
            "1 : many",
            "An assignment can have zero or more conversion events.",
        ],
        [
            "Assignment",
            "GuardrailObservation",
            "1 : many",
            "Operational outcomes are recorded against assignments.",
        ],
        [
            "Experiment",
            "MetricDefinition",
            "1 : many",
            "Versioned metric definitions preserve changes over time.",
        ],
        [
            "Experiment",
            "GuardrailMetric",
            "1 : many",
            "Each experiment can define multiple operational limits.",
        ],
        [
            "Experiment",
            "ExperimentResult",
            "1 : many",
            "Each analysis run creates a traceable result snapshot.",
        ],
    ], columns=[
        "Parent",
        "Child",
        "Cardinality",
        "Meaning",
    ])

    display_dataframe(relationships)

    st.markdown("### Business rules and assumptions")

    st.markdown(
        """
        - Primary analysis is **intention-to-treat**.
        - Conversion outcome window: **30 days** after assignment.
        - SRM is compared with the experiment's **configured allocation**.
        - Missing telemetry and noncompliance remain visible.
        - Guardrail observations are persisted as source records.
        - Ground-truth labels are stored separately from analytical records.
        - All records are synthetic and generated from a fixed seed.
        """
    )

    st.markdown("### Important limitations")

    st.warning(
        "This is a hackathon prototype using synthetic data. "
        "It demonstrates workflow and evaluation logic, not a "
        "production experimentation service. Production deployment "
        "would need authentication, role-based access, database "
        "migrations, audit logs, background jobs and a reviewed "
        "statistical analysis plan."
    )