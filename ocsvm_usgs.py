import pandas as pd

from dataretrieval import waterdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# USGS monitoring station ID.
# Make sure this ID matches the location you want to study.
SITE = "USGS-03431500"

# USGS parameter code 00060 means streamflow:
# the amount of water flowing through the river.
PARAMETER = "00060"
STATISTIC = "00003"  # Daily average streamflow

# The date range of historical data to download.
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# 1. Download historical streamflow measurements from USGS.
print("Downloading USGS data...")

data, metadata = waterdata.get_daily(
    monitoring_location_id=SITE,
    parameter_code=PARAMETER,
    statistic_id=STATISTIC,
    time=f"{START_DATE}/{END_DATE}",
)

# Stop the program if USGS did not return any data.
if data.empty:
    raise ValueError(
        "No data returned. Check the site, parameter, "
        "and date range."
    )

print("Columns returned:", data.columns.tolist())

# 2. Clean the downloaded data.
# Convert dates and measurements into usable formats.
data["time"] = pd.to_datetime(
    data["time"], utc=True, errors="coerce"
)
data["value"] = pd.to_numeric(
    data["value"], errors="coerce"
)

# Remove rows with missing or invalid dates/measurements.
# Put observations in date order and remove duplicate dates.
data = (
    data.dropna(subset=["time", "value"])
    .sort_values("time")
    .drop_duplicates(subset=["time"])
    .reset_index(drop=True)
)

# Stop if too few observations remain for this experiment.
if len(data) < 100:
    raise ValueError(
        "Not enough observations for this initial experiment."
    )

# 3. Add information about the time of year.
# River flow can change with the seasons, so these features
# help the model account for recurring yearly patterns.
# Sine and cosine represent the day of the year as numbers.
day = data["time"].dt.dayofyear
data["day_sin"] = __import__("numpy").sin(
    2 * __import__("numpy").pi * day / 365.25
)
data["day_cos"] = __import__("numpy").cos(
    2 * __import__("numpy").pi * day / 365.25
)

# Give the model streamflow and seasonal information.
features = ["value", "day_sin", "day_cos"]
X = data[features]

# 4. Split the data by date.
# Use the earliest 80% to teach the model what is typical.
# Keep the latest 20% separate to test it on later data.
split = int(len(data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

# Keep the dates and original measurements for the test period.
test_data = data.iloc[split:].copy()

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training ends:", data["time"].iloc[split - 1])
print("Testing begins:", data["time"].iloc[split])

# 5. Create the One-Class SVM anomaly-detection model.
# StandardScaler puts the features on comparable scales.
# One-Class SVM learns a boundary around patterns it considers
# normal and can flag observations outside that boundary.
#
# The RBF kernel allows the model to learn curved boundaries.
# nu=0.03 sets a parameter related to the expected fraction
# of training errors; it does not guarantee 3% of test data
# will be flagged as anomalies.
model = make_pipeline(
    StandardScaler(),
    OneClassSVM(
        kernel="rbf",
        nu=0.03,
        gamma="scale",
    ),
)

# Teach the model using ONLY the earlier training data.
# The later test data is not used during training.
model.fit(X_train)

# 6. Check the later data for unusual measurements.
# +1 means the model considers an observation normal.
# -1 means the model flags it as a possible anomaly.
test_data["prediction"] = model.predict(X_test)

# The decision score indicates how far an observation is
# from the model's learned boundary.
# Lower, more negative scores generally mean more unusual.
test_data["anomaly_score"] = model.decision_function(X_test)

# Mark each observation that the model flagged as unusual.
test_data["is_anomaly"] = (
    test_data["prediction"] == -1
)

# 7. Display the flagged observations and save the results.
anomalies = test_data[test_data["is_anomaly"]]

print("\nTest-period observations:", len(test_data))
print("Flagged anomalies:", len(anomalies))

# Show up to 20 flagged observations with the lowest scores.
print("\nMost unusual observations:")
print(
    anomalies[
        ["time", "value", "anomaly_score"]
    ].sort_values("anomaly_score").head(20).to_string(
        index=False
    )
)

# Save all test-period observations, predictions, and scores
# to a CSV file so the results can be inspected later.
test_data.to_csv("ocsvm_test_results.csv", index=False)
print("\nSaved results to ocsvm_test_results.csv")
