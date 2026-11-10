
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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import DB_PATH, SEED
from src.generator import generate_dataset
from src.storage import save_frames, load_frames, export_bundle
from src.analytics import analyze_experiment, all_experiment_summaries
from src.validation import validate_frames, integrity_summary


# =============================================================
# PAGE CONFIGURATION AND DESIGN
# =============================================================

st.set_page_config(
    page_title="ExperimentIQ | Decision Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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
    background: linear-gradient(115deg,#14213d,#254b70);
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
    border: 1px solid rgba(128,145,170,.25);
    border-radius: 13px;
    padding: 13px 15px;
    background: rgba(128,145,170,.045);
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(128,145,170,.2);
    border-radius: 10px;
}
div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(128,145,170,.15);
    border-radius: 12px;
    padding: 5px;
}
</style>
""", unsafe_allow_html=True)


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def ground_truth_evaluation(frames, issues):
    truth = frames.get("ground_truth", pd.DataFrame())

    if truth.empty:
        return pd.DataFrame(), 0, 0

    detected = (
        set(issues.issue_type.astype(str))
        if not issues.empty
        else set()
    )

    rows = []

    for _, item in truth.iterrows():
        found = item.issue_type in detected

        rows.append({
            "Scenario": item.issue_type.replace("_", " ").title(),
            "Expected decision": item.expected_decision.replace(
                "_", " "
            ).title(),
            "Detected": "Yes" if found else "No",
            "Status": "PASS" if found else "REVIEW",
        })

    result = pd.DataFrame(rows)

    return (
        result,
        int(result.Status.eq("PASS").sum()),
        len(result),
    )


def fmt_pct(value):
    return f"{float(value):+.2f}%"


def metric_help(name):
    return {
        "Conversion rate": (
            "Share of assigned users with at least one valid "
            "conversion inside the 30-day outcome window."
        ),
        "Relative uplift": (
            "Percentage change in treatment conversion rate "
            "relative to control."
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
    }.get(name, "")


def analytical_assignments(frames, experiment_id):
    """Return assignment rows after applying analytics eligibility
    and conflicting-variant exclusion rules.
    """
    assignments = frames["assignments"].copy()
    audience = frames["audience_members"]
    variants = frames["variants"]

    a = assignments[
        assignments.experiment_id == experiment_id
    ].copy()

    if "is_eligible" in a.columns:
        a = a[a["is_eligible"].fillna(False).astype(bool)]

    elif "eligible" in a.columns:
        a = a[a["eligible"].fillna(False).astype(bool)]

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

        a = a[a["eligible"].fillna(False).astype(bool)]

    conflict_counts = (
        a.groupby("user_id")["variant_id"].nunique()
    )

    conflicting_users = conflict_counts[
        conflict_counts > 1
    ].index

    a = a[~a.user_id.isin(conflicting_users)].copy()

    a = a.merge(
        variants[["variant_id", "variant_key", "name"]],
        on="variant_id",
        how="left",
        validate="many_to_one",
    )

    return a


def make_funnel_chart(assigned, exposed, converted):
    funnel = pd.DataFrame({
        "Stage": ["Assigned", "Exposed", "Converted"],
        "Users": [assigned, exposed, converted],
    })

    fig = px.funnel(
        funnel,
        x="Users",
        y="Stage",
        title="Experiment conversion funnel",
    )

    fig.update_layout(
        height=370,
        margin=dict(l=15, r=15, t=55, b=20),
        showlegend=False,
    )

    return fig


def make_effect_plot(summaries):
    """Show relative uplift and confidence intervals when available."""
    data = summaries.copy()

    required = {"name", "uplift_pct"}

    if not required.issubset(data.columns):
        return None

    has_ci = {"uplift_ci_low", "uplift_ci_high"}.issubset(
        data.columns
    )

    if has_ci:
        data = data.sort_values("uplift_pct")

        fig = go.Figure()

        for _, row in data.iterrows():
            low = row["uplift_ci_low"]
            high = row["uplift_ci_high"]

            if pd.notna(low) and pd.notna(high):
                fig.add_trace(go.Scatter(
                    x=[low, high],
                    y=[row["name"], row["name"]],
                    mode="lines",
                    line=dict(width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ))

        fig.add_trace(go.Scatter(
            x=data["uplift_pct"],
            y=data["name"],
            mode="markers",
            marker=dict(size=11),
            name="Relative uplift",
        ))

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
        )

        fig.update_layout(
            title="Treatment effect across experiments",
            xaxis_title="Relative uplift (%)",
            yaxis_title="",
            height=440,
            margin=dict(l=15, r=15, t=55, b=35),
        )

        return fig

    fig = px.bar(
        data.sort_values("uplift_pct"),
        x="uplift_pct",
        y="name",
        orientation="h",
        color="decision" if "decision" in data.columns else None,
        title="Relative treatment uplift by experiment",
        labels={
            "uplift_pct": "Relative uplift (%)",
            "name": "Experiment",
        },
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        height=440,
        margin=dict(l=15, r=15, t=55, b=30),
    )

    return fig


def show_chart(fig):
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


# =============================================================
# LOAD PERSISTED DATA
# =============================================================

if "frames" not in st.session_state:
    try:
        st.session_state.frames = (
            load_frames(DB_PATH) if DB_PATH.exists() else None
        )
    except Exception as exc:
        st.session_state.frames = None
        st.error(f"Could not load the saved dataset: {exc}")


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
    st.caption(f"Fixed seed · {SEED}")

    with st.expander("Dataset controls"):
        user_count = st.select_slider(
            "Audience size",
            options=[1000, 5000, 10000, 25000, 50000],
            value=50000,
        )

        seed = st.number_input(
            "Random seed",
            value=SEED,
            step=1,
        )

        if st.button(
            "Generate dataset",
            type="primary",
            use_container_width=True,
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
            "Generating a new dataset replaces source tables; "
            "versioned experiment-result snapshots are retained. "
            "Export source data first if you need to keep an older dataset."
        )


# =============================================================
# HERO
# =============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">Experimentation & Decision Intelligence</div>
    <h1>ExperimentIQ</h1>
    <p>
        Design experiments, validate data quality, understand
        treatment effects and make evidence-led decisions.
    </p>
</div>
""", unsafe_allow_html=True)


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
        "Set a hypothesis, primary metric, audience and allocation."
    )

    y.markdown(
        "**2 · Validate**\n\n"
        "Check assignment quality, event order, telemetry and guardrails."
    )

    z.markdown(
        "**3 · Decide**\n\n"
        "Estimate treatment effect, uncertainty and rollout risks."
    )

    if st.button("Create the demo dataset", type="primary"):
        with st.spinner("Generating linked audience and event records..."):
            generated = generate_dataset(50000, 10, SEED)
            save_frames(generated, DB_PATH)
            st.session_state.frames = load_frames(DB_PATH)

        st.rerun()

    st.stop()


# =============================================================
# DATA PREPARATION AND CACHING
# =============================================================

experiments = frames["experiments"]
assignments = frames["assignments"]
conversions = frames["conversion_events"]
exposures = frames["exposure_events"]
variants = frames["variants"]

if st.session_state.get("derived") is None:
    with st.spinner(
        "Preparing dashboard summaries and data-quality checks..."
    ):
        st.session_state.derived = {
            "issues": validate_frames(frames),
            "summaries": all_experiment_summaries(frames),
            "integrity": integrity_summary(frames),
        }

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
    experiments.experiment_id.tolist(),
    format_func=lambda x: (
        f"{x} · {exp_lookup.loc[x, 'name']}"
    ),
)

selected_exp = exp_lookup.loc[selected]

# Pass the actual generated observation and metric-definition records.
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

clean_assignments = analytical_assignments(frames, selected)


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

    k1.metric(
        "Experiments",
        f"{len(experiments):,}",
    )

    k2.metric(
        "Eligible audience",
        f"{int(frames['audience_members'].eligible.sum()):,}",
    )

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

    # Portfolio uplift chart
    st.markdown("### Experiment performance")

    left, right = st.columns([1.6, 1])

    with left:
        fig = px.bar(
            summaries,
            x="name",
            y="uplift_pct",
            color="decision",
            hover_data=[
                c for c in ["p_value", "srm_status"]
                if c in summaries.columns
            ],
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

    with right:
        counts = (
            summaries.decision.value_counts()
            .rename_axis("Decision")
            .reset_index(name="Experiments")
        )

        show_chart(
            px.pie(
                counts,
                names="Decision",
                values="Experiments",
                hole=0.58,
                title="Portfolio decision distribution",
            ).update_layout(height=400)
        )

    st.markdown("### Treatment-effect comparison")

    show_chart(make_effect_plot(summaries))

    st.markdown("### Evidence and quality")

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Known scenarios detected",
        f"{truth_pass}/{truth_total}",
    )

    q2.metric(
        "High-severity findings",
        int((issues.severity == "HIGH").sum())
        if not issues.empty else 0,
    )

    q3.metric(
        "Database checks passed",
        f"{int(integrity.status.eq('PASS').sum())}/"
        f"{len(integrity)}",
    )

    st.progress(truth_pass / max(truth_total, 1))

    st.caption(
        "Ground-truth labels are stored separately from analytical inputs."
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

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
    )

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

    st.dataframe(
        experiments.rename(columns={
            "experiment_id": "Experiment ID",
            "name": "Name",
            "hypothesis": "Hypothesis",
            "primary_metric": "Primary metric",
            "status": "Lifecycle status",
            "allocation_a": "Control allocation",
            "allocation_b": "Treatment allocation",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Selected experiment design")

    c1, c2, c3 = st.columns(3)

    c1.metric("Lifecycle", selected_exp.status.title())

    c2.metric(
        "Primary metric",
        selected_exp.primary_metric.replace("_", " ").title(),
    )

    c3.metric(
        "Planned split",
        f"{selected_exp.allocation_a:.0%} / "
        f"{selected_exp.allocation_b:.0%}",
    )

    st.markdown("**Hypothesis**")
    st.info(selected_exp.hypothesis)

    st.markdown("**Variants**")

    st.dataframe(
        variants[variants.experiment_id == selected],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Metric definition history**")

    st.dataframe(
        frames["metric_definitions"].query(
            "experiment_id == @selected"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The current prototype reads experiment definitions from "
        "generated seed data. Interactive authoring is a future extension."
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
        assignments.experiment_id == selected
    ].merge(
        variants[["variant_id", "variant_key", "name"]],
        on="variant_id",
        how="left",
    )

    counts = (
        clean_assignments.variant_key
        .value_counts()
        .reindex(["A", "B"], fill_value=0)
    )

    raw_conflicts = (
        raw.groupby("user_id")["variant_id"].nunique() > 1
    ).sum()

    ca, cb, cc, cd = st.columns(4)

    ca.metric("Raw assignments", f"{len(raw):,}")

    cb.metric(
        "Valid control",
        f"{counts['A']:,}",
    )

    cc.metric(
        "Valid treatment",
        f"{counts['B']:,}",
    )

    cd.metric(
        "Conflicting users",
        f"{raw_conflicts:,}",
    )

    st.markdown("### Observed versus expected allocation")

    chart = pd.DataFrame({
        "Variant": ["Control", "Treatment"],
        "Observed users": [
            counts["A"],
            counts["B"],
        ],
        "Expected users": [
            len(clean_assignments)
            * result["expected_allocation_a"],
            len(clean_assignments)
            * result["expected_allocation_b"],
        ],
    })

    fig = px.bar(
        chart,
        x="Variant",
        y=["Observed users", "Expected users"],
        barmode="group",
        title="Valid observed versus expected assignment",
    )

    fig.update_layout(height=390)

    show_chart(fig)

    st.markdown("### Allocation proportions")

    observed = pd.DataFrame({
        "Variant": ["Control", "Treatment"],
        "Observed share": [
            counts["A"] / max(len(clean_assignments), 1),
            counts["B"] / max(len(clean_assignments), 1),
        ],
        "Configured share": [
            result["expected_allocation_a"],
            result["expected_allocation_b"],
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
        f"{result['srm_p_value']:.4g}",
    )

    y.metric(
        "SRM status",
        result["srm_status"],
    )

    if result["srm_status"] == "FAIL":
        st.error(
            "Sample-ratio mismatch detected. Investigate allocation "
            "before trusting the treatment estimate."
        )
    else:
        st.success(
            "No sample-ratio mismatch detected at the configured threshold."
        )

    st.markdown("### Assignment exception records")

    subset = issues[
        issues.issue_type.isin([
            "MULTIPLE_VARIANT_ASSIGNMENT",
            "SAMPLE_RATIO_MISMATCH",
        ])
    ]

    st.dataframe(
        subset,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View raw assignment records"):
        st.dataframe(
            raw.head(500),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================
# 4. USER ACTIVITY
# =============================================================

elif page == "User Activity":

    st.subheader("User activity & event tracking")

    st.caption(
        "Follow the journey from assignment to exposure and conversion."
    )

    e = exposures[
        exposures.experiment_id == selected
    ].copy()

    c = conversions[
        conversions.experiment_id == selected
    ].copy()

    a = clean_assignments.copy()

    valid_ids = set(a.assignment_id.astype(str))

    e = e[
        e.assignment_id.astype(str).isin(valid_ids)
    ]

    c = c[
        c.assignment_id.astype(str).isin(valid_ids)
    ]

    assigned_count = a.user_id.nunique()
    exposed_count = e.assignment_id.nunique()
    converted_count = c.assignment_id.nunique()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Assigned", f"{assigned_count:,}")
    m2.metric("Exposed", f"{exposed_count:,}")

    m3.metric(
        "No recorded exposure",
        f"{max(assigned_count - exposed_count, 0):,}",
    )

    m4.metric("Converted assignments", f"{converted_count:,}")

    left, right = st.columns(2)

    with left:
        show_chart(
            make_funnel_chart(
                assigned_count,
                exposed_count,
                converted_count,
            )
        )

    with right:
        if (
            not e.empty
            and "telemetry_status" in e.columns
        ):
            status = (
                e.telemetry_status.value_counts()
                .rename_axis("Telemetry")
                .reset_index(name="Events")
            )

            show_chart(
                px.pie(
                    status,
                    names="Telemetry",
                    values="Events",
                    hole=0.55,
                    title="Exposure telemetry completeness",
                ).update_layout(height=370)
            )

    st.caption(
        "Non-exposure is treated as noncompliance. The primary "
        "analysis remains intention-to-treat, based on original assignment."
    )

    with st.expander("Recent exposure events"):
        st.dataframe(
            e.head(250),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Recent conversion events"):
        st.dataframe(
            c.head(250),
            use_container_width=True,
            hide_index=True,
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

    v = result["variants"].set_index("variant")

    ca, cb, cc = st.columns(3)

    ca.metric(
        "Control conversion",
        f"{v.loc['A', 'conversion_rate']:.2%}",
    )

    cb.metric(
        "Treatment conversion",
        f"{v.loc['B', 'conversion_rate']:.2%}",
    )

    cc.metric(
        "Relative uplift",
        fmt_pct(result["uplift_pct"]),
        help=metric_help("Relative uplift"),
    )

    st.markdown("### Conversion rate by variant")

    rates = result["variants"].copy()

    rates["Conversion rate (%)"] = (
        rates.conversion_rate * 100
    )

    rates["Upper error"] = (
        rates.ci_high - rates.conversion_rate
    ).clip(lower=0) * 100

    rates["Lower error"] = (
        rates.conversion_rate - rates.ci_low
    ).clip(lower=0) * 100

    fig = px.bar(
        rates,
        x="name",
        y="Conversion rate (%)",
        color="name",
        title="Conversion rate with 95% Wilson intervals",
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

    st.markdown("### Treatment effect and uncertainty")

    x, y, z = st.columns(3)

    x.metric(
        "P-value",
        f"{result['p_value']:.4g}",
        help=metric_help("P-value"),
    )

    y.metric(
        "95% CI · absolute difference",
        f"[{result['ci_low']:.2%}, {result['ci_high']:.2%}]",
        help=metric_help("95% confidence interval"),
    )

    z.metric(
        "Randomization",
        result["srm_status"],
    )

    st.markdown("### Decision recommendation")

    decision = str(result["decision"]).upper()

    st.markdown(f"## {result['decision']}")

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

    if not guardrails.empty:
        st.dataframe(
            guardrails.rename(columns={
                "metric": "Guardrail",
                "observed": "Observed value",
                "threshold": "Allowed threshold",
                "status": "Status",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Visualize treatment and control observations when
        # source guardrail observations are available.
        observations = frames.get(
            "guardrail_observations",
            pd.DataFrame(),
        )

        if (
            not observations.empty
            and "assignment_id" in observations.columns
        ):
            assigned = clean_assignments[
                ["assignment_id", "variant_key"]
            ]

            go_data = observations.merge(
                assigned,
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

                guardrail_plot = pd.DataFrame(plot_rows)

                show_chart(
                    px.bar(
                        guardrail_plot,
                        x="Guardrail",
                        y="Observed rate (%)",
                        color="Variant",
                        barmode="group",
                        title="Observed guardrail rates by variant",
                    ).update_layout(height=370)
                )

    else:
        st.info("No guardrail results are available for this experiment.")

    st.caption(
        "Guardrail indicators use generated synthetic observations. "
        "Statistical estimates are intended for demonstration, not "
        "production causal inference."
    )

    st.markdown("### Stored result snapshots")

    st.dataframe(
        frames["experiment_results"].query(
            "experiment_id == @selected"
        ),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================
# 6. DATA QUALITY
# =============================================================

elif page == "Data Quality":

    st.subheader("Data quality & exception queue")

    st.caption(
        "Review detected exceptions, severity and known-scenario coverage."
    )

    high = (
        int((issues.severity == "HIGH").sum())
        if not issues.empty else 0
    )

    medium = (
        int((issues.severity == "MEDIUM").sum())
        if not issues.empty else 0
    )

    low = (
        int((issues.severity == "LOW").sum())
        if not issues.empty else 0
    )

    x, y, z, w = st.columns(4)

    x.metric("Total findings", f"{len(issues):,}")
    y.metric("High severity", f"{high:,}")
    z.metric("Medium severity", f"{medium:,}")
    w.metric("Low severity", f"{low:,}")

    if not issues.empty:

        left, right = st.columns(2)

        with left:
            sev = (
                issues.severity.value_counts()
                .rename_axis("Severity")
                .reset_index(name="Findings")
            )

            show_chart(
                px.bar(
                    sev,
                    x="Severity",
                    y="Findings",
                    color="Severity",
                    title="Findings by severity",
                ).update_layout(height=350)
            )

        with right:
            issue_counts = (
                issues.issue_type.value_counts()
                .rename_axis("Issue type")
                .reset_index(name="Findings")
                .sort_values("Findings")
            )

            show_chart(
                px.bar(
                    issue_counts,
                    x="Findings",
                    y="Issue type",
                    orientation="h",
                    title="Most frequent data-quality issues",
                ).update_layout(height=350)
            )

        f1, f2 = st.columns(2)

        with f1:
            sev_filter = st.multiselect(
                "Severity",
                ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"],
            )

        with f2:
            type_filter = st.multiselect(
                "Issue type",
                sorted(issues.issue_type.unique()),
                default=sorted(issues.issue_type.unique()),
            )

        filtered = issues[
            issues.severity.isin(sev_filter)
            & issues.issue_type.isin(type_filter)
        ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.success("No data-quality findings detected.")

    st.markdown("### Known-scenario evaluation")

    st.metric(
        "Ground-truth detection coverage",
        f"{truth_pass}/{truth_total}",
    )

    if truth_total:
        st.progress(truth_pass / truth_total)

    st.dataframe(
        truth_eval,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Ground-truth labels are not included in analytical inputs. "
        "Detection coverage is category-level in this demonstration."
    )

    with st.expander("Database integrity checks"):
        st.dataframe(
            integrity,
            use_container_width=True,
            hide_index=True,
        )


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
    ]

    inv = pd.DataFrame([
        {
            "Table": table,
            "Rows": len(frames[table]),
            "Columns": len(frames[table].columns),
        }
        for table in table_names
    ])

    st.dataframe(
        inv,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Preview a source table")

    chosen = st.selectbox("Table", table_names)

    st.dataframe(
        frames[chosen].head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Export complete dataset")

    if st.button(
        "Prepare CSV + JSON export bundle",
        type="primary",
    ):
        try:
            with st.spinner("Preparing export bundle and SHA-256 manifest..."):
                with tempfile.TemporaryDirectory() as tmp:
                    export_bundle(frames, tmp)

                    buf = io.BytesIO()

                    with zipfile.ZipFile(
                        buf,
                        "w",
                        zipfile.ZIP_DEFLATED,
                    ) as zf:
                        for path in Path(tmp).iterdir():
                            if path.is_file():
                                zf.write(path, path.name)

                    st.session_state.export_zip = buf.getvalue()

        except Exception as exc:
            st.error(f"Export failed: {exc}")

    if st.session_state.get("export_zip"):
        st.download_button(
            "Download export bundle (.zip)",
            data=st.session_state.export_zip,
            file_name="ExperimentIQ_exports.zip",
            mime="application/zip",
        )

    st.warning(
        "Ground truth is exported as its own file. "
        "Do not merge it into model or analysis input features."
    )

    st.markdown("### Reproducibility settings")

    st.code(
        f"Seed: {int(experiments.seed.iloc[0])}\n"
        f"Experiments: {len(experiments)}\n"
        f"Audience rows: {len(frames['audience_members']):,}",
        language="text",
    )

    if st.button("Verify fixed-seed reproducibility"):

        try:
            with st.spinner(
                "Generating the same dataset twice and comparing SHA-256 checksums..."
            ):
                run_a = generate_dataset(
                    len(frames["audience_members"]),
                    len(experiments),
                    int(experiments.seed.iloc[0]),
                )

                run_b = generate_dataset(
                    len(frames["audience_members"]),
                    len(experiments),
                    int(experiments.seed.iloc[0]),
                )

                reproducibility_rows = []

                for table in run_a:
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
            st.error(f"Reproducibility check failed: {exc}")

    if st.session_state.get("reproducibility") is not None:
        rep = st.session_state.reproducibility

        st.dataframe(
            rep,
            use_container_width=True,
            hide_index=True,
        )

        if rep["Checksum match"].all():
            st.success(
                "Reproducibility verified: all generated tables match."
            )
        else:
            st.error(
                "A checksum mismatch was found. Review affected tables."
            )

    st.markdown("### Integrity summary")

    st.dataframe(
        integrity,
        use_container_width=True,
        hide_index=True,
    )


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
        ("Customer / prospect", "Receives a control or treatment experience."),
        ("Marketing / sales ops", "Defines and launches an experiment."),
        ("Support agent", "Helps resolve user or experience issues."),
        ("Analyst", "Checks evidence and interprets results."),
        ("Administrator", "Maintains data and system configuration."),
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
        ["Experiment", "Variant", "1 : many",
         "Each experiment has control and treatment variants."],
        ["Experiment", "Assignment", "1 : many",
         "An experiment has many user assignments."],
        ["AudienceMember", "Assignment", "1 : many over time",
         "A user may participate in multiple experiments."],
        ["Variant", "Assignment", "1 : many",
         "Each assignment references one variant."],
        ["Assignment", "ExposureEvent", "1 : many",
         "An assignment can have zero or more exposure events."],
        ["Assignment", "ConversionEvent", "1 : many",
         "An assignment can have zero or more conversion events."],
        ["Assignment", "GuardrailObservation", "1 : many",
         "Operational outcomes are recorded against assignments."],
        ["Experiment", "MetricDefinition", "1 : many",
         "Versioned metric definitions preserve changes over time."],
        ["Experiment", "GuardrailMetric", "1 : many",
         "Each experiment can define multiple operational limits."],
        ["Experiment", "ExperimentResult", "1 : many",
         "Each analysis run creates a traceable result snapshot."],
    ], columns=[
        "Parent",
        "Child",
        "Cardinality",
        "Meaning",
    ])

    st.dataframe(
        relationships,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Business rules and assumptions")

    st.markdown("""
    - Primary analysis is **intention-to-treat**.
    - Conversion outcome window: **30 days** after assignment.
    - SRM is compared with the experiment's **configured allocation**.
    - Missing telemetry and noncompliance remain visible.
    - Guardrail observations are persisted as source records.
    - Ground-truth labels are stored separately from analytical records.
    - All records are synthetic and generated from a fixed seed.
    """)

    st.markdown("### Important limitations")

    st.warning(
        "This is a hackathon prototype using synthetic data. "
        "It demonstrates workflow and evaluation logic, not a "
        "production experimentation service. Production deployment "
        "would need authentication, role-based access, database "
        "migrations, audit logs, background jobs and a reviewed "
        "statistical analysis plan."
    )
