import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium

from risk_model import (
    load_risk_data,
    get_ranked_locations,
    get_unmanned_high_risk
)

from allocation import (
    allocate_officers,
    compare_deployment
)

from simulation import (
    simulate_incident
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Traffic Risk Control Room",
    page_icon="🚦",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title(
    "🚦 AI Traffic Risk Control Room"
)

st.subheader(
    "Nagpur Smart Traffic Risk & Police Deployment System"
)


# ==========================================
# LOAD DATA
# ==========================================

df = load_risk_data()


# ==========================================
# ALLOCATE OFFICERS
# ==========================================

df = allocate_officers(
    df,
    total_officers=20
)

df = compare_deployment(df)


# ==========================================
# DASHBOARD METRICS
# ==========================================

high_risk = len(
    df[df["risk_category"] == "HIGH"]
)

medium_risk = len(
    df[df["risk_category"] == "MEDIUM"]
)

low_risk = len(
    df[df["risk_category"] == "LOW"]
)

unmanned_high = len(
    get_unmanned_high_risk(df)
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Junctions",
    len(df)
)

col2.metric(
    "🔴 High Risk",
    high_risk
)

col3.metric(
    "🟠 Medium Risk",
    medium_risk
)

col4.metric(
    "⚠️ Unmanned High Risk",
    unmanned_high
)


# ==========================================
# HEATMAP
# ==========================================

st.header("🗺️ Traffic Risk Heatmap")


m = folium.Map(
    location=[
        21.1458,
        79.0882
    ],
    zoom_start=11
)


for _, row in df.iterrows():

    if row["risk_category"] == "HIGH":

        color = "red"

    elif row["risk_category"] == "MEDIUM":

        color = "orange"

    else:

        color = "green"


    popup_text = f"""
    <b>Junction:</b>
    {row['CCTV Junction No.']}<br>

    <b>Risk Score:</b>
    {row['risk_score']}<br>

    <b>Risk:</b>
    {row['risk_category']}<br>

    <b>Officers:</b>
    {row['officer_present']}<br>

    <b>Reason:</b>
    {row['risk_reason']}
    """


    folium.CircleMarker(

        location=[
            row["latitude"],
            row["longitude"]
        ],

        radius=7,

        color=color,

        fill=True,

        popup=folium.Popup(
            popup_text,
            max_width=350
        )
    ).add_to(m)


st_folium(
    m,
    width=1100,
    height=550
)


# ==========================================
# RANKED LOCATIONS
# ==========================================

st.header(
    "🚨 Locations Requiring Police Attention"
)


ranked = get_ranked_locations(df)


display_columns = [
    "CCTV Junction No.",
    "location",
    "risk_score",
    "risk_category",
    "officer_present",
    "recommended_officers",
    "risk_reason"
]


st.dataframe(
    ranked[
        display_columns
    ].head(15),
    use_container_width=True
)


# ==========================================
# OFFICER ALLOCATION
# ==========================================

st.header(
    "👮 Officer Allocation"
)


allocation_columns = [
    "CCTV Junction No.",
    "risk_score",
    "officer_present",
    "recommended_officers",
    "officer_change"
]


st.dataframe(
    ranked[
        allocation_columns
    ].head(20),
    use_container_width=True
)


# ==========================================
# UNMANNED HIGH-RISK
# ==========================================

st.header(
    "⚠️ High-Risk Locations Currently Unmanned"
)


unmanned = get_unmanned_high_risk(df)


st.dataframe(
    unmanned[
        [
            "CCTV Junction No.",
            "risk_score",
            "risk_category",
            "risk_reason"
        ]
    ],
    use_container_width=True
)


# ==========================================
# INCIDENT SIMULATION
# ==========================================

st.header(
    "🚨 Dynamic Incident Simulation"
)


selected_junction = st.selectbox(
    "Select Junction",
    df["CCTV Junction No."]
)


incident_type = st.selectbox(
    "Select Incident",
    [
        "Major Accident",
        "Heavy Congestion",
        "Crowd Formation",
        "Road Blockage"
    ]
)


if st.button(
    "🚨 Simulate Incident"
):

    simulated_df, incident = simulate_incident(
        df,
        selected_junction,
        incident_type
    )


    if incident is not None:

        st.error(
            f"Incident detected at Junction "
            f"{incident['junction']}"
        )

        st.write(
            f"Incident Type: "
            f"**{incident['incident']}**"
        )

        st.write(
            f"Risk increased from "
            f"**{incident['old_score']:.2f}** "
            f"to "
            f"**{incident['new_score']:.2f}**"
        )

        st.warning(
            "AI recommends immediate police redeployment."
        )


# ==========================================
# MANUAL OVERRIDE
# ==========================================

st.header(
    "🎛️ Manual Officer Override"
)


override_junction = st.selectbox(
    "Select Junction for Override",
    df["CCTV Junction No."],
    key="override"
)


override_count = st.number_input(
    "Number of Officers",
    min_value=0,
    max_value=10,
    value=1
)


if st.button(
    "Apply Manual Override"
):

    st.success(
        f"Manual override applied: "
        f"{override_count} officers assigned "
        f"to Junction {override_junction}"
    )


# ==========================================
# BASELINE VS AI
# ==========================================

st.header(
    "📊 Baseline vs Recommended Deployment"
)


baseline_unmanned = len(
    df[
        (df["risk_category"] == "HIGH") &
        (df["officer_present"] == 0)
    ]
)


recommended_unmanned = len(
    df[
        (df["risk_category"] == "HIGH") &
        (df["recommended_officers"] == 0)
    ]
)


col1, col2 = st.columns(2)


col1.metric(
    "Baseline High-Risk Unmanned",
    baseline_unmanned
)


col2.metric(
    "Recommended High-Risk Unmanned",
    recommended_unmanned
)