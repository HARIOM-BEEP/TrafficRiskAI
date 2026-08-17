def simulate_incident(
    df,
    junction_number,
    incident_type="Major Accident"
):

    result = df.copy()

    # Find selected junction
    index_list = result[
        result["CCTV Junction No."] == junction_number
    ].index

    # Junction not found
    if len(index_list) == 0:
        return result, None

    index = index_list[0]

    # Current risk
    old_score = float(
        result.loc[index, "risk_score"]
    )

    # Determine risk increase
    if incident_type == "Major Accident":
        increase = 30

    elif incident_type == "Heavy Congestion":
        increase = 20

    elif incident_type == "Crowd Formation":
        increase = 25

    elif incident_type == "Road Blockage":
        increase = 15

    else:
        increase = 20

    # New risk
    new_score = min(
        100,
        old_score + increase
    )

    # Update risk
    result.loc[index, "risk_score"] = new_score

    result.loc[index, "risk_category"] = "HIGH"

    # Incident causes heavy congestion
    result.loc[index, "congestion_level"] = 10

    # Return incident information
    incident = {
        "junction": junction_number,
        "incident": incident_type,
        "old_score": old_score,
        "new_score": new_score
    }

    return result, incident