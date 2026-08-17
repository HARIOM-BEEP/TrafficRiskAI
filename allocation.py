import pandas as pd


def allocate_officers(df, total_officers=20):
    result = df.copy()

    # Calculate normalized scores (ensure scale is 0.0 to 1.0)
    def normalize_score(score):
        s = float(score)
        return s / 100.0 if s > 1.0 else s

    result["normalized_risk"] = result["risk_score"].apply(normalize_score)

    # Unmanned locations get additional priority
    # Priority = Risk * (1 + 0.5 * isUnmanned)
    result["priority_score"] = result.apply(
        lambda row: row["normalized_risk"] * (1.0 + 0.5 * (1.0 if int(row.get("officer_present", 0)) == 0 else 0.0)),
        axis=1
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

        score = row["normalized_risk"]
        cat = str(row.get("risk_category", "")).upper()

        if cat == "HIGH" or cat == "CRITICAL" or score >= 0.70:
            officers = 2
        elif cat == "MEDIUM" or score >= 0.40:
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