import warnings
warnings.filterwarnings("ignore")

from supabase import create_client
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import joblib
import os

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_table(table_name, batch_size=1000):
    all_data = []
    start = 0
    while True:
        end = start + batch_size - 1
        res = supabase.table(table_name).select("*").range(start, end).execute()
        data = res.data or []
        if not data:
            break
        all_data.extend(data)
        if len(data) < batch_size:
            break
        start += batch_size
    df = pd.DataFrame(all_data)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df

def main():
    rate_df = fetch_table("rate_and_classification")
    pop_df = fetch_table("population_records")
    weather_df = fetch_table("weather_records")

    merged = rate_df.merge(
        pop_df[["Barangay", "Year", "Week", "Population"]],
        on=["Barangay", "Year", "Week"],
        how="left"
    ).merge(
        weather_df[[
            "Year", "Week",
            "average_weekly_temperature",
            "average_weekly_relative_humidity",
            "total_weekly_rainfall",
            "average_weekly_wind_speed",
            "average_weekly_wind_direction"
        ]],
        on=["Year", "Week"],
        how="left"
    )

    citywide_df = (
        merged.groupby(["Year", "Week"])
        .agg({
            "Cases": "sum",
            "Population": "sum",
            "average_weekly_temperature": "mean",
            "average_weekly_relative_humidity": "mean",
            "total_weekly_rainfall": "mean",
            "average_weekly_wind_speed": "mean",
            "average_weekly_wind_direction": "mean"
        })
        .reset_index()
    )
    
    citywide_df["attack_rate"] = (citywide_df["Cases"] / citywide_df["Population"]) * 100000

    mean_rate = citywide_df["attack_rate"].mean()
    std_rate = citywide_df["attack_rate"].std()

    def classify_risk(rate, mean, std):
        if rate > mean + 2 * std:
            return "High Risk"
        elif rate > mean + std:
            return "Moderate Risk"
        else:
            return "Low Risk"

    citywide_df["risk_classification"] = citywide_df["attack_rate"].apply(
        lambda r: classify_risk(r, mean_rate, std_rate)
    )

    X = citywide_df[[
        "Year", "Week", "Population",
        "average_weekly_temperature",
        "average_weekly_relative_humidity",
        "total_weekly_rainfall",
        "average_weekly_wind_speed",
        "average_weekly_wind_direction"
    ]]
    y = citywide_df["risk_classification"]

    le_risk = LabelEncoder()
    y_encoded = le_risk.fit_transform(y)

    class_counts = Counter(y_encoded)
    print("Class distribution:", class_counts)

    # --- If any class has < 2 samples, skip stratification ---
    if min(class_counts.values()) < 2:
        print(" Some classes have fewer than 2 samples. Using non-stratified split.")
        stratify_train = None
        stratify_val = None
    else:
        stratify_train = y_encoded
        stratify_val = y_encoded

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_train
    )

    temp_counts = Counter(y_temp)
    if len(temp_counts) < len(class_counts) or min(temp_counts.values()) < 2:
        print(" Validation/test set too small for stratification. Using regular split.")
        stratify_val = None
    else:
        stratify_val = y_temp

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=stratify_val
    )

    n_neighbors = 3
    if min(class_counts.values()) >= n_neighbors:
        smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    else:
        print("Skipping SMOTE due to insufficient samples in one or more classes.")
        X_balanced, y_balanced = X_train, y_train

    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(le_risk.classes_),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(
        X_balanced, y_balanced,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    os.makedirs("forecast", exist_ok=True)
    joblib.dump(model, "forecast/model_citywide.joblib")
    joblib.dump(le_risk, "forecast/le_risk_citywide.joblib")

    print("Citywide model training completed and saved successfully.")

if __name__ == "__main__":
    main()