import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD ORIGINAL CCTV DATA
# ==========================================

INPUT_FILE = "data/CCTV_Junctionslist-Nagpur-2017-2018.csv"
OUTPUT_FILE = "data/traffic_risk_data.csv"

df = pd.read_csv(INPUT_FILE, encoding="cp1252")

print("Original dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. GROUP CCTV POLES INTO JUNCTIONS
# ==========================================

# Find useful columns
junction_col = "CCTV Junction No."
pole_col = "CCTV Pole No."
location_col = "CCTV LOCATION"

# Group multiple CCTV poles belonging to same junction
junctions = df.groupby(junction_col).agg({
    pole_col: "count",
    location_col: "first"
}).reset_index()

junctions.rename(
    columns={
        pole_col: "camera_count",
        location_col: "location"
    },
    inplace=True
)

print("\nNumber of junctions:", len(junctions))


# ==========================================
# 3. GENERATE APPROXIMATE GPS COORDINATES
# ==========================================

np.random.seed(42)

n = len(junctions)

# Approximate Nagpur region
junctions["latitude"] = (
    21.10 + np.random.uniform(0, 0.12, n)
)

junctions["longitude"] = (
    79.00 + np.random.uniform(0, 0.15, n)
)


# ==========================================
# 4. GENERATE TRAFFIC FEATURES
# ==========================================

junctions["traffic_volume"] = np.random.randint(
    500, 5000, n
)

junctions["average_speed"] = np.random.randint(
    15, 70, n
)

junctions["accident_count"] = np.random.poisson(
    3, n
)

junctions["pedestrian_density"] = np.random.randint(
    10, 100, n
)

junctions["congestion_level"] = np.random.randint(
    1, 11, n
)

junctions["weather_risk"] = np.random.randint(
    0, 6, n
)

junctions["night_time_risk"] = np.random.randint(
    0, 6, n
)


# ==========================================
# 5. CURRENT POLICE DEPLOYMENT
# ==========================================

junctions["officer_present"] = np.random.choice(
    [0, 1],
    size=n,
    p=[0.7, 0.3]
)


# ==========================================
# 6. SPEED RISK
# ==========================================

# Assume approximately 40 km/h is desirable
junctions["speed_risk"] = abs(
    junctions["average_speed"] - 40
)


# ==========================================
# 7. NORMALIZATION
# ==========================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return 0

    return (series - minimum) / (maximum - minimum)


junctions["traffic_volume_norm"] = normalize(
    junctions["traffic_volume"]
)

junctions["accident_count_norm"] = normalize(
    junctions["accident_count"]
)

junctions["pedestrian_density_norm"] = normalize(
    junctions["pedestrian_density"]
)

junctions["congestion_level_norm"] = normalize(
    junctions["congestion_level"]
)

junctions["weather_risk_norm"] = normalize(
    junctions["weather_risk"]
)

junctions["night_time_risk_norm"] = normalize(
    junctions["night_time_risk"]
)

junctions["speed_risk_norm"] = normalize(
    junctions["speed_risk"]
)


# ==========================================
# 8. RISK SCORE
# ==========================================

junctions["risk_score"] = (
    30 * junctions["accident_count_norm"] +
    20 * junctions["traffic_volume_norm"] +
    15 * junctions["congestion_level_norm"] +
    15 * junctions["pedestrian_density_norm"] +
    10 * junctions["speed_risk_norm"] +
    5 * junctions["weather_risk_norm"] +
    5 * junctions["night_time_risk_norm"]
)

junctions["risk_score"] = (
    junctions["risk_score"].round(2)
)


# ==========================================
# 9. RISK CATEGORY
# ==========================================

def risk_category(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


junctions["risk_category"] = (
    junctions["risk_score"].apply(risk_category)
)


# ==========================================
# 10. EXPLAINABLE RISK
# ==========================================

def explain_risk(row):

    reasons = []

    if row["accident_count_norm"] > 0.7:
        reasons.append("High incident risk")

    if row["traffic_volume_norm"] > 0.7:
        reasons.append("Heavy traffic volume")

    if row["congestion_level_norm"] > 0.7:
        reasons.append("High congestion")

    if row["pedestrian_density_norm"] > 0.7:
        reasons.append("High pedestrian density")

    if row["speed_risk_norm"] > 0.7:
        reasons.append("Unsafe speed pattern")

    if row["weather_risk_norm"] > 0.7:
        reasons.append("Weather-related risk")

    if len(reasons) == 0:
        reasons.append("No major risk factors")

    return ", ".join(reasons)


junctions["risk_reason"] = junctions.apply(
    explain_risk,
    axis=1
)


# ==========================================
# 11. SAVE DATASET
# ==========================================

junctions.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("DATASET CREATED SUCCESSFULLY")
print("================================")

print("Output:", OUTPUT_FILE)
print("Rows:", len(junctions))

print("\nRisk distribution:")
print(junctions["risk_category"].value_counts())

print("\nTop 10 risky junctions:")

print(
    junctions[
        [
            junction_col,
            "risk_score",
            "risk_category",
            "officer_present"
        ]
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(10)
)