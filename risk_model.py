import pandas as pd


def load_risk_data():

    return pd.read_csv(
        "data/traffic_risk_data.csv"
    )


def get_ranked_locations(df):

    return df.sort_values(
        "risk_score",
        ascending=False
    )


def get_high_risk_locations(df):

    return df[
        df["risk_category"].isin(["HIGH", "CRITICAL"])
    ]


def get_unmanned_high_risk(df):

    return df[
        df["risk_category"].isin(["HIGH", "CRITICAL"]) &
        (df["officer_present"] == 0)
    ]


def get_risk_explanation(row):

    return row["risk_reason"]