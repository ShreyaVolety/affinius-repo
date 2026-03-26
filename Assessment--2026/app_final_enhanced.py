
import json
import os

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


st.set_page_config(
    page_title="US Migration Dashboard",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 1.3rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(16,185,129,0.08));
        border: 1px solid rgba(120,120,120,0.18);
        padding: 0.8rem;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    template="plotly_white",
    height=470,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


@st.cache_data
def load_data():
    pair_df = pd.read_csv("pair_interactions.csv")
    inflow_df = pd.read_csv("inflow.csv")
    outflow_df = pd.read_csv("outflow.csv")
    county_df = pd.read_csv("flows_combined.csv")
    od_df = pd.read_csv("all_counties_flows.csv")
    enrichment_df = pd.read_csv("county_enrichment_2003_2004.csv")

    # Standardize FIPS
    for df, col in [
        (pair_df, "county1"), (pair_df, "county2"),
        (inflow_df, "FIPS"), (outflow_df, "FIPS"),
        (county_df, "FIPS"), (county_df, "fips"),
        (od_df, "from_FIPS"), (od_df, "to_FIPS"),
        (enrichment_df, "fips"),
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(".0", "", regex=False).str.zfill(5)

 

    # Numeric cleanup
    numeric_cols = [
        (pair_df, ["Total migration flow"]),
        (inflow_df, ["Inflows"]),
        (outflow_df, ["Outflows"]),
        (county_df, ["Inflows", "Outflows", "Net Flow",
                    "median_hh_income_2004", "poverty_rate_pct_2004",
                    "median_home_value_2000", "population_2004"]),
        (od_df, ["Returns", "Exemptions", "Agg_Adj_gross_income"]),
        (enrichment_df, ["median_hh_income_2004", "poverty_rate_pct_2004",
                        "median_home_value_2000", "population_2004"]),
    ]

    for df, cols in numeric_cols:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean names
    if "county_name" in county_df.columns:
        county_df["county_name"] = county_df["county_name"].astype(str).str.strip()
    if "County Name" in county_df.columns:
        county_df["County Name"] = county_df["County Name"].astype(str).str.strip()
    if "urban_rural" in county_df.columns:
        county_df["urban_rural"] = county_df["urban_rural"].fillna("Unknown").astype(str).str.title()

    county_df["County Label"] = county_df["county_name"].fillna(county_df.get("County Name", "")).astype(str)
    if "State Name" in county_df.columns:
        county_df["County Label"] = county_df["County Label"] + ", " + county_df["State Name"].fillna("").astype(str).str.title()
    county_df["Flow Balance"] = np.where(county_df["Net Flow"] >= 0, "Net Positive", "Net Negative")

    # Pair label
    pair_df["county1"] = pair_df["county1"].astype(str).str.zfill(5)
    pair_df["county2"] = pair_df["county2"].astype(str).str.zfill(5)
    pair_df["Pair Label"] = (
        pair_df["County1 name"].astype(str) + ", " + pair_df["State1 name"].astype(str)
        + " ↔ " +
        pair_df["County2 name"].astype(str) + ", " + pair_df["State2 name"].astype(str)
    )

    # Merge county type / labels into OD flows
    lookup = county_df[["FIPS", "County Label", "urban_rural"]].drop_duplicates()
    lookup_from = lookup.rename(columns={"FIPS": "from_FIPS", "County Label": "From Label", "urban_rural": "From Type"})
    lookup_to = lookup.rename(columns={"FIPS": "to_FIPS", "County Label": "To Label", "urban_rural": "To Type"})
    od_df = od_df.merge(lookup_from, on="from_FIPS", how="left").merge(lookup_to, on="to_FIPS", how="left")
    od_df["Movement Type"] = od_df["From Type"].fillna("Unknown") + " → " + od_df["To Type"].fillna("Unknown")

    return pair_df, inflow_df, outflow_df, county_df, od_df, enrichment_df


def get_ai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def format_hover_fields(df: pd.DataFrame, fields: list[str]) -> dict:
    hover = {}
    for field in fields:
        if field in df.columns:
            hover[field] = ":,.0f" if pd.api.types.is_numeric_dtype(df[field]) else True
    return hover


def top_n_bar_plot(df: pd.DataFrame, value_col: str, label_col: str, title: str, n: int = 15, color_scale="Blues"):
    plot_df = df.nlargest(n, value_col).sort_values(value_col, ascending=True).copy()
    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation="h",
        text=value_col,
        color=value_col,
        color_continuous_scale=color_scale,
        title=title,
        hover_data=format_hover_fields(plot_df, ["Inflows", "Outflows", "Net Flow"]),
    )
    fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, yaxis_title="", xaxis_title=value_col)
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)


def flow_scatter_plot(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df["Inflows + 1"] = plot_df["Inflows"] + 1
    plot_df["Outflows + 1"] = plot_df["Outflows"] + 1
    plot_df["Abs Net"] = plot_df["Net Flow"].abs()
    fig = px.scatter(
        plot_df,
        x="Inflows + 1",
        y="Outflows + 1",
        color="Flow Balance",
        size="Abs Net",
        size_max=22,
        hover_name="County Label",
        hover_data={
            "Inflows": ":,.0f",
            "Outflows": ":,.0f",
            "Net Flow": ":,.0f",
            "Inflows + 1": False,
            "Outflows + 1": False,
            "Abs Net": False,
        },
        log_x=True,
        log_y=True,
        opacity=0.72,
        title="County Inflows vs Outflows",
    )
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(title="Inflows + 1 (log scale)")
    fig.update_yaxes(title="Outflows + 1 (log scale)")
    st.plotly_chart(fig, use_container_width=True)


def histogram_plot(df, x, title, color=None, nbins=60, marginal=None):
    fig = px.histogram(
        df,
        x=x,
        color=color,
        nbins=nbins,
        barmode="overlay" if color else "relative",
        opacity=0.7,
        marginal=marginal,
        title=title,
    )
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def box_plot_by_type(df, value_col, title):
    fig = px.box(
        df.dropna(subset=[value_col, "urban_rural"]),
        x="urban_rural",
        y=value_col,
        color="urban_rural",
        points=False,
        category_orders={"urban_rural": ["Rural", "Suburban", "Urban"]},
        title=title,
    )
    fig.update_layout(**PLOT_LAYOUT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def county_profile_metrics(selected_row: pd.Series):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("County Type", selected_row.get("urban_rural", "N/A"))
    income = selected_row.get("median_hh_income_2004", np.nan)
    poverty = selected_row.get("poverty_rate_pct_2004", np.nan)
    home = selected_row.get("median_home_value_2000", np.nan)
    c2.metric("Median HH Income (2004)", "N/A" if pd.isna(income) else f"${income:,.0f}")
    c3.metric("Poverty Rate (2004)", "N/A" if pd.isna(poverty) else f"{poverty:.1f}%")
    c4.metric("Median Home Value (2000)", "N/A" if pd.isna(home) else f"${home:,.0f}")


def county_directional_views(od_df: pd.DataFrame, selected_fips: str):
    incoming = (
        od_df[od_df["to_FIPS"] == selected_fips]
        .groupby(["from_FIPS", "From Label"], as_index=False)[["Exemptions", "Returns"]]
        .sum()
        .sort_values("Exemptions", ascending=False)
    )
    outgoing = (
        od_df[od_df["from_FIPS"] == selected_fips]
        .groupby(["to_FIPS", "To Label"], as_index=False)[["Exemptions", "Returns"]]
        .sum()
        .sort_values("Exemptions", ascending=False)
    )
    incoming = incoming.rename(columns={"From Label": "Peer Label", "from_FIPS": "Peer FIPS"})
    outgoing = outgoing.rename(columns={"To Label": "Peer Label", "to_FIPS": "Peer FIPS"})
    return incoming, outgoing


def county_link_bar(df: pd.DataFrame, flow_col: str, label_col: str, title: str, color_scale: str):
    plot_df = df.head(10).sort_values(flow_col, ascending=True).copy()
    fig = px.bar(
        plot_df,
        x=flow_col,
        y=label_col,
        orientation="h",
        text=flow_col,
        color=flow_col,
        color_continuous_scale=color_scale,
        title=title,
        hover_data={flow_col: ":,.0f"},
    )
    fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, yaxis_title="", xaxis_title="Migrants (Exemptions)")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)


def movement_mix_for_selected_pair(od_df: pd.DataFrame, fips1: str, fips2: str) -> pd.DataFrame:
    subset = od_df[
        ((od_df["from_FIPS"] == fips1) & (od_df["to_FIPS"] == fips2)) |
        ((od_df["from_FIPS"] == fips2) & (od_df["to_FIPS"] == fips1))
    ].copy()
    if subset.empty:
        return subset
    mix = subset.groupby("Movement Type", as_index=False)["Exemptions"].sum().sort_values("Exemptions", ascending=False)
    return mix


def pair_direction_table(pair_df: pd.DataFrame, county_df: pd.DataFrame) -> pd.DataFrame:
    lookup = county_df[["FIPS", "urban_rural"]].drop_duplicates()
    pair_view = pair_df.merge(lookup.rename(columns={"FIPS": "county1", "urban_rural": "County1 Type"}), on="county1", how="left")
    pair_view = pair_view.merge(lookup.rename(columns={"FIPS": "county2", "urban_rural": "County2 Type"}), on="county2", how="left")
    pair_view["Interaction Type"] = pair_view["County1 Type"].fillna("Unknown") + " ↔ " + pair_view["County2 Type"].fillna("Unknown")
    return pair_view


def skew_explanation(series: pd.Series, label: str) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return f"No usable data for {label.lower()}."
    skew = s.skew()
    mean = s.mean()
    median = s.median()
    if skew > 1:
        direction = "strongly right-skewed"
        reason = (
            "a small number of very large counties dominate total movement, "
            "while most counties have much smaller flows"
        )
    elif skew > 0.3:
        direction = "moderately right-skewed"
        reason = "larger counties still pull the mean above the median"
    elif skew < -1:
        direction = "strongly left-skewed"
        reason = "a few very negative values create a long left tail"
    elif skew < -0.3:
        direction = "moderately left-skewed"
        reason = "negative tail counties pull the mean below the median"
    else:
        direction = "roughly symmetric"
        reason = "the mean and median are relatively similar"
    return (
        f"**{label}** is {direction} (skew = {skew:.2f}). "
        f"The mean is {mean:,.0f} and the median is {median:,.0f}, suggesting {reason}."
    )


def generate_ai_answer(question: str, pair_df: pd.DataFrame, county_df: pd.DataFrame, od_df: pd.DataFrame) -> str:
    client = get_ai_client()
    if client is None:
        return (
            "AI feature is not active. Add OPENAI_API_KEY to your environment or Streamlit secrets "
            "and install the openai package to enable natural-language answers."
        )

    top_pairs = pair_df.nlargest(12, "Total migration flow")[
        ["Pair Label", "Total migration flow"]
    ].to_dict(orient="records")
    top_inflows = county_df.nlargest(12, "Inflows")[
        ["County Label", "Inflows", "Outflows", "Net Flow", "urban_rural"]
    ].to_dict(orient="records")
    top_outflows = county_df.nlargest(12, "Outflows")[
        ["County Label", "Inflows", "Outflows", "Net Flow", "urban_rural"]
    ].to_dict(orient="records")
    movement_types = od_df.groupby("Movement Type", as_index=False)["Exemptions"].sum().sort_values("Exemptions", ascending=False).head(10)
    movement_types = movement_types.to_dict(orient="records")

    prompt = f"""
You are helping explain a US county migration dashboard.

Answer the user's question using ONLY the structured data below.
If the answer is not supported by the data, say so clearly.
Be concise and analytical.

Top county pairs:
{json.dumps(top_pairs, indent=2)}

Top inflow counties:
{json.dumps(top_inflows, indent=2)}

Top outflow counties:
{json.dumps(top_outflows, indent=2)}

Top movement types between county classes:
{json.dumps(movement_types, indent=2)}

User question:
{question}
""".strip()

    response = client.responses.create(model="gpt-5-mini", input=prompt)
    return response.output_text


pair_df, inflow_df, outflow_df, county_df, od_df, enrichment_df = load_data()
pair_view = pair_direction_table(pair_df, county_df)

st.title("US Migration Patterns Dashboard")
st.caption("County migration patterns, directional county-to-county flows, and enrichment context for 2003–2004.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "County Explorer", "Pair Explorer", "Distributions & Insights", "Ask the Data"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("County pairs", f"{len(pair_df):,}")
    c2.metric("Counties", f"{county_df['FIPS'].nunique():,}")
    c3.metric("Directional OD records", f"{len(od_df):,}")
    c4.metric("Net-positive counties", f"{(county_df['Net Flow'] > 0).sum():,}")

    left, right = st.columns(2)
    with left:
        top_n_bar_plot(county_df, "Inflows", "County Label", "Top Counties by In-Migration", n=15, color_scale="Blues")
    with right:
        top_n_bar_plot(county_df, "Outflows", "County Label", "Top Counties by Out-Migration", n=15, color_scale="Reds")

    st.subheader("Inflow vs Outflow")
    st.caption("Bubble size reflects absolute net flow. Both axes are log-scaled because county flows are highly skewed.")
    flow_scatter_plot(county_df)

    left, right = st.columns(2)
    with left:
        movement_by_type = od_df.groupby("Movement Type", as_index=False)["Exemptions"].sum().sort_values("Exemptions", ascending=False)
        top_n_bar_plot(movement_by_type.rename(columns={"Exemptions": "Migrants"}), "Migrants", "Movement Type", "Migration by County-Type Transition", n=9, color_scale="Tealgrn")
    with right:
        type_net = county_df.groupby("urban_rural", as_index=False)[["Inflows", "Outflows", "Net Flow"]].sum()
        fig = px.bar(
            type_net.melt(id_vars="urban_rural", value_vars=["Inflows", "Outflows", "Net Flow"], var_name="Metric", value_name="Value"),
            x="urban_rural",
            y="Value",
            color="Metric",
            barmode="group",
            title="Aggregate Flows by County Type",
            category_orders={"urban_rural": ["Rural", "Suburban", "Urban"]},
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("County Explorer")
    state_options = ["All"] + sorted(county_df["State Name"].dropna().astype(str).str.title().unique().tolist())
    selected_state = st.selectbox("Filter by state", state_options)

    county_view = county_df.copy()
    county_view["State Display"] = county_view["State Name"].astype(str).str.title()
    if selected_state != "All":
        county_view = county_view[county_view["State Display"] == selected_state]

    selected_label = st.selectbox("Choose a county", sorted(county_view["County Label"].dropna().unique()))
    selected_row = county_view[county_view["County Label"] == selected_label].iloc[0]
    selected_fips = selected_row["FIPS"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Inflows", f"{int(selected_row['Inflows']):,}")
    c2.metric("Outflows", f"{int(selected_row['Outflows']):,}")
    c3.metric("Net Flow", f"{int(selected_row['Net Flow']):,}")

    st.markdown("### County Context")
    county_profile_metrics(selected_row)

    st.markdown("### Directional Migration Flows")
    incoming_df, outgoing_df = county_directional_views(od_df, selected_fips)
    left, right = st.columns(2)
    with left:
        county_link_bar(incoming_df, "Exemptions", "Peer Label", f"Top Counties Sending Migrants to {selected_label}", "Blues")
    with right:
        county_link_bar(outgoing_df, "Exemptions", "Peer Label", f"Top Counties {selected_label} Is Losing Migrants To", "Reds")

    st.markdown("### Selected County Flow Mix")
    selected_mix = pd.DataFrame({
        "Metric": ["Inflows", "Outflows", "Net Flow"],
        "Value": [selected_row["Inflows"], selected_row["Outflows"], selected_row["Net Flow"]],
    })
    fig = px.bar(
        selected_mix,
        x="Metric",
        y="Value",
        text="Value",
        color="Metric",
        title=f"Flow Profile: {selected_label}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(texttemplate="%{y:,.0f}")
    fig.update_layout(**PLOT_LAYOUT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Pair Explorer")
    top_k = st.slider("How many top pairs to show?", 5, 30, 15)
    top_pair_df = pair_view.nlargest(top_k, "Total migration flow").copy()

    left, right = st.columns([1.15, 0.85])
    with left:
        fig = px.bar(
            top_pair_df.sort_values("Total migration flow", ascending=True),
            x="Total migration flow",
            y="Pair Label",
            orientation="h",
            text="Total migration flow",
            color="Interaction Type",
            title="Top County Pair Interactions",
            hover_data={"Total migration flow": ":,.0f", "Interaction Type": True},
        )
        fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
        fig.update_layout(**PLOT_LAYOUT, yaxis_title="", xaxis_title="Total migration flow")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        interaction_mix = pair_view.groupby("Interaction Type", as_index=False)["Total migration flow"].sum().sort_values("Total migration flow", ascending=False)
        fig = px.pie(
            interaction_mix,
            values="Total migration flow",
            names="Interaction Type",
            title="County-Pair Interaction Mix by Region Type",
            hole=0.45,
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Pair Movement Direction")
    selected_pair = st.selectbox("Choose a pair to inspect directionality", pair_view.sort_values("Total migration flow", ascending=False)["Pair Label"].unique())
    pair_row = pair_view[pair_view["Pair Label"] == selected_pair].iloc[0]
    pair_mix = movement_mix_for_selected_pair(od_df, pair_row["county1"], pair_row["county2"])

    if pair_mix.empty:
        st.info("No directional flow records found for this pair.")
    else:
        fig = px.bar(
            pair_mix.sort_values("Exemptions", ascending=True),
            x="Exemptions",
            y="Movement Type",
            orientation="h",
            text="Exemptions",
            color="Movement Type",
            title=f"Directional Movement Types Within {selected_pair}",
        )
        fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
        fig.update_layout(**PLOT_LAYOUT, yaxis_title="", xaxis_title="Migrants (Exemptions)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pair_view.sort_values("Total migration flow", ascending=False)[
            ["Pair Label", "Total migration flow", "County1 Type", "County2 Type", "Interaction Type"]
        ],
        use_container_width=True,
        height=420,
    )

with tab4:
    st.subheader("Major Insights: Distributions and Skew")

    c1, c2, c3 = st.columns(3)
    c1.markdown(skew_explanation(county_df["Inflows"], "Inflows"))
    c2.markdown(skew_explanation(county_df["Outflows"], "Outflows"))
    c3.markdown(skew_explanation(county_df["Net Flow"], "Net Flow"))

    left, right = st.columns(2)
    with left:
        histogram_plot(county_df, "Inflows", "Distribution of County Inflows", nbins=70)
    with right:
        histogram_plot(county_df, "Outflows", "Distribution of County Outflows", nbins=70)

    left, right = st.columns(2)
    with left:
        histogram_plot(county_df, "Net Flow", "Distribution of Net Flows", nbins=70)
    with right:
        fig = px.violin(
            county_df,
            x="urban_rural",
            y="Net Flow",
            color="urban_rural",
            box=True,
            points=False,
            category_orders={"urban_rural": ["Rural", "Suburban", "Urban"]},
            title="Net Flow Distribution by County Type",
        )
        fig.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Inflow and Outflow Distributions by County Type")
    left, right = st.columns(2)
    with left:
        histogram_plot(
            county_df[county_df["urban_rural"].isin(["Urban", "Suburban", "Rural"])],
            "Inflows",
            "Inflows by Urban / Suburban / Rural County",
            color="urban_rural",
            nbins=60,
        )
    with right:
        histogram_plot(
            county_df[county_df["urban_rural"].isin(["Urban", "Suburban", "Rural"])],
            "Outflows",
            "Outflows by Urban / Suburban / Rural County",
            color="urban_rural",
            nbins=60,
        )

    left, right = st.columns(2)
    with left:
        box_plot_by_type(county_df, "Inflows", "Inflows by County Type (Box Plot)")
    with right:
        box_plot_by_type(county_df, "Outflows", "Outflows by County Type (Box Plot)")

    st.info(
        "Why the skew? Migration flows are concentrated in a relatively small number of large metro counties. "
        "That creates long right tails for inflows and outflows. Net flow is more centered near zero, but still "
        "shows heavy tails because a few counties experience very large net gains or losses."
    )

with tab5:
    st.subheader("Ask the Data")
    st.write("Example questions:")
    st.write("- Which counties dominate inflows?")
    st.write("- Are the biggest outflow counties also big inflow counties?")
    st.write("- What region-type transitions dominate movement?")
    st.write("- Are urban–suburban interactions more important than rural–rural interactions?")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask a question about the migration data...")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        answer = generate_ai_answer(user_prompt, pair_view, county_df, od_df)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
