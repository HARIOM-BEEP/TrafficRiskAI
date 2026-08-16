import pandas as pd


def allocate_officers(df, total_officers=20):

    result = df.copy()

    # Unmanned locations get additional priority
    result["priority_score"] = (
        result["risk_score"] *
        (
            1 +
            0.5 *
            (1 - result["officer_present"])
        )
    )

    # Start with zero recommended officers
    result["recommended_officers"] = 0

    # Highest priority first
    result = result.sort_values(
        "priority_score",
        ascending=False
    )

    officers_remaining = total_officers

    # First allocation
    for index, row in result.iterrows():

        if officers_remaining <= 0:
            break

        if row["risk_score"] >= 80:
            officers = 2

        elif row["risk_score"] >= 60:
            officers = 1

        else:
            officers = 0

        officers = min(
            officers,
            officers_remaining
        )

        result.loc[
            index,
            "recommended_officers"
        ] = officers

        officers_remaining -= officers

    return result


def compare_deployment(df):

    comparison = df.copy()

    comparison["officer_change"] = (
        comparison["recommended_officers"]
        -
        comparison["officer_present"]
    )

    return comparison