import json
import os

import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


st.set_page_config(
    page_title="US Migration Dashboard",
    page_icon="🧭",
    layout="wide"
)


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
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
    height=480,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


@st.cache_data
def load_data():
    pair_df = pd.read_csv("pair_interactions.csv", dtype=str)
    inflow_df = pd.read_csv("inflow.csv", dtype=str)
    outflow_df = pd.read_csv("outflow.csv", dtype=str)

    if "Total migration flow" in pair_df.columns:
        pair_df["Total migration flow"] = pd.to_numeric(
            pair_df["Total migration flow"], errors="coerce"
        )

    if "Inflows" in inflow_df.columns:
        inflow_df["Inflows"] = pd.to_numeric(inflow_df["Inflows"], errors="coerce")

    if "Outflows" in outflow_df.columns:
        outflow_df["Outflows"] = pd.to_numeric(outflow_df["Outflows"], errors="coerce")

    for col in ["FIPS", "county1", "county2", "state1", "state2"]:
        if col in pair_df.columns:
            pair_df[col] = pair_df[col].astype(str).str.zfill(5 if "county" in col else 2)

    if "FIPS" in inflow_df.columns:
        inflow_df["FIPS"] = inflow_df["FIPS"].astype(str).str.zfill(5)

    if "FIPS" in outflow_df.columns:
        outflow_df["FIPS"] = outflow_df["FIPS"].astype(str).str.zfill(5)

    return pair_df, inflow_df, outflow_df


def build_county_flow_table(inflow_df: pd.DataFrame, outflow_df: pd.DataFrame) -> pd.DataFrame:
    county_flow = inflow_df[["FIPS", "County Name", "State Name", "Inflows"]].copy()

    county_flow = county_flow.merge(
        outflow_df[["FIPS", "Outflows"]],
        on="FIPS",
        how="outer"
    )

    if "County Name" not in county_flow.columns or county_flow["County Name"].isna().any():
        county_flow = county_flow.merge(
            outflow_df[["FIPS", "County Name", "State Name"]],
            on="FIPS",
            how="left",
            suffixes=("", "_out")
        )
        county_flow["County Name"] = county_flow["County Name"].fillna(county_flow.get("County Name_out"))
        county_flow["State Name"] = county_flow["State Name"].fillna(county_flow.get("State Name_out"))
        county_flow = county_flow.drop(
            columns=[c for c in ["County Name_out", "State Name_out"] if c in county_flow.columns]
        )

    county_flow["Inflows"] = pd.to_numeric(county_flow["Inflows"], errors="coerce").fillna(0)
    county_flow["Outflows"] = pd.to_numeric(county_flow["Outflows"], errors="coerce").fillna(0)
    county_flow["Net Migration"] = county_flow["Inflows"] - county_flow["Outflows"]
    county_flow["County Label"] = (
        county_flow["County Name"].fillna("Unknown") + ", " + county_flow["State Name"].fillna("")
    )
    county_flow["Flow Balance"] = county_flow["Net Migration"].apply(
        lambda x: "Net Positive" if x >= 0 else "Net Negative"
    )
    return county_flow


def format_hover_fields(df: pd.DataFrame, fields: list[str]) -> dict:
    hover_data = {}
    for field in fields:
        if field in df.columns:
            hover_data[field] = ":,.0f" if pd.api.types.is_numeric_dtype(df[field]) else True
    return hover_data


def top_n_bar_plot(df: pd.DataFrame, value_col: str, label_col: str, title: str, n: int = 15):
    plot_df = df.nlargest(n, value_col).sort_values(value_col, ascending=True).copy()
    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation="h",
        text=value_col,
        color=value_col,
        color_continuous_scale="Blues",
        title=title,
        hover_data=format_hover_fields(plot_df, ["Inflows", "Outflows", "Net Migration"]),
    )
    fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, yaxis_title="", xaxis_title=value_col)
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)


def flow_scatter_plot(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df["Inflows + 1"] = plot_df["Inflows"] + 1
    plot_df["Outflows + 1"] = plot_df["Outflows"] + 1
    plot_df["Abs Net Migration"] = plot_df["Net Migration"].abs()

    fig = px.scatter(
        plot_df,
        x="Inflows + 1",
        y="Outflows + 1",
        color="Flow Balance",
        size="Abs Net Migration",
        size_max=20,
        hover_name="County Label",
        hover_data={
            "Inflows": ":,.0f",
            "Outflows": ":,.0f",
            "Net Migration": ":,.0f",
            "Inflows + 1": False,
            "Outflows + 1": False,
            "Abs Net Migration": False,
        },
        log_x=True,
        log_y=True,
        title="County Inflows vs Outflows",
        opacity=0.75,
    )
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(title="Inflows + 1 (log scale)")
    fig.update_yaxes(title="Outflows + 1 (log scale)")
    st.plotly_chart(fig, use_container_width=True)


def pair_bar_plot(pair_view: pd.DataFrame, top_k: int):
    top_pairs_view = pair_view.nlargest(top_k, "Total migration flow").sort_values(
        "Total migration flow", ascending=True
    )
    fig = px.bar(
        top_pairs_view,
        x="Total migration flow",
        y="Pair Label",
        orientation="h",
        text="Total migration flow",
        color="Total migration flow",
        color_continuous_scale="Tealgrn",
        title="Top County Pairs by Total Interaction",
        hover_data=format_hover_fields(top_pairs_view, ["Total migration flow"]),
    )
    fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, yaxis_title="", xaxis_title="Total migration flow")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)


def get_ai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def generate_ai_answer(question: str, pair_df: pd.DataFrame, county_flow: pd.DataFrame) -> str:
    client = get_ai_client()
    if client is None:
        return (
            "AI feature is not active. Add OPENAI_API_KEY to Streamlit secrets or your environment "
            "and install the openai package to enable natural-language insights."
        )

    top_pairs = pair_df.nlargest(15, "Total migration flow")[
        [c for c in [
            "pair_key", "County1 name", "State1 name", "County2 name", "State2 name", "Total migration flow"
        ] if c in pair_df.columns]
    ].to_dict(orient="records")

    top_inflows = county_flow.nlargest(15, "Inflows")[
        ["County Label", "Inflows", "Outflows", "Net Migration"]
    ].to_dict(orient="records")

    top_outflows = county_flow.nlargest(15, "Outflows")[
        ["County Label", "Inflows", "Outflows", "Net Migration"]
    ].to_dict(orient="records")

    prompt = f"""
You are helping explain a US county migration dashboard.

Answer the user's question using ONLY the structured data below.
If the answer is not supported by the data, say so clearly.
Be concise and analytical.

Available data:
TOP COUNTY PAIRS:
{json.dumps(top_pairs, indent=2)}

TOP INFLOW COUNTIES:
{json.dumps(top_inflows, indent=2)}

TOP OUTFLOW COUNTIES:
{json.dumps(top_outflows, indent=2)}

User question:
{question}
""".strip()

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    return response.output_text


pair_df, inflow_df, outflow_df = load_data()
county_flow = build_county_flow_table(inflow_df, outflow_df)

st.title("US Migration Patterns Dashboard")
st.caption("Interactive dashboard built from county migration flow outputs.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "County Explorer", "Pair Explorer", "Ask the Data"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("County pairs", f"{len(pair_df):,}")
    c2.metric("Counties with inflow data", f"{inflow_df['FIPS'].nunique():,}")
    c3.metric("Counties with outflow data", f"{outflow_df['FIPS'].nunique():,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Top In-Migration Counties")
        top_n_bar_plot(county_flow, "Inflows", "County Label", "Top Counties by In-Migration", n=15)
    with right:
        st.subheader("Top Out-Migration Counties")
        top_n_bar_plot(county_flow, "Outflows", "County Label", "Top Counties by Out-Migration", n=15)

    st.subheader("Inflow vs Outflow")
    st.caption("Bubble size reflects the absolute value of net migration.")
    flow_scatter_plot(county_flow)

with tab2:
    st.subheader("County Explorer")
    state_options = ["All"] + sorted([s for s in county_flow["State Name"].dropna().unique()])
    selected_state = st.selectbox("Filter by state", state_options)

    county_view = county_flow.copy()
    if selected_state != "All":
        county_view = county_view[county_view["State Name"] == selected_state]

    selected_label = st.selectbox(
        "Choose a county",
        sorted(county_view["County Label"].dropna().unique())
    )

    selected_row = county_view[county_view["County Label"] == selected_label].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Inflows", f"{int(selected_row['Inflows']):,}")
    c2.metric("Outflows", f"{int(selected_row['Outflows']):,}")
    c3.metric("Net Migration", f"{int(selected_row['Net Migration']):,}")

    st.plotly_chart(
        px.bar(
            pd.DataFrame({
                "Metric": ["Inflows", "Outflows", "Net Migration"],
                "Value": [
                    selected_row["Inflows"],
                    selected_row["Outflows"],
                    selected_row["Net Migration"],
                ],
            }),
            x="Metric",
            y="Value",
            text="Value",
            color="Metric",
            title=f"Flow Profile: {selected_label}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        ).update_traces(texttemplate="%{y:,.0f}").update_layout(**PLOT_LAYOUT, showlegend=False),
        use_container_width=True,
    )

    st.dataframe(
        county_view.sort_values("Net Migration", ascending=False)[
            ["County Label", "Inflows", "Outflows", "Net Migration"]
        ],
        use_container_width=True,
        height=450,
    )

with tab3:
    st.subheader("Top County Pair Interactions")

    pair_view = pair_df.copy()
    if set(["County1 name", "State1 name", "County2 name", "State2 name"]).issubset(pair_view.columns):
        pair_view["Pair Label"] = (
            pair_view["County1 name"] + ", " + pair_view["State1 name"] +
            " ↔ " +
            pair_view["County2 name"] + ", " + pair_view["State2 name"]
        )
    else:
        pair_view["Pair Label"] = pair_view.get("pair_key", pair_view.index.astype(str))

    top_k = st.slider("How many top pairs to show?", 5, 30, 15)
    pair_bar_plot(pair_view, top_k)

    st.dataframe(
        pair_view.sort_values("Total migration flow", ascending=False),
        use_container_width=True,
        height=450,
    )

with tab4:
    st.subheader("Ask the Data")
    st.write("Example questions:")
    st.write("- Which counties dominate inflows?")
    st.write("- Are the biggest outflow counties also big inflow counties?")
    st.write("- What pattern do the top pair interactions suggest?")

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

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_ai_answer(user_prompt, pair_df, county_flow)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
