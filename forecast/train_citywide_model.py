import warnings
warnings.filterwarnings("ignore")

from supabase import create_client
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
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
        merged.groupby(["Year", "Month", "Week"])
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
    
    citywide_df["attack_rate"] = (citywide_df["Cases"] / citywide_df["Population"]) * 10000

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
    label_map = {"Low Risk":0, "Moderate Risk":1, "High Risk":2}
    citywide_df['risk_code'] = citywide_df['risk_classification'].map(label_map)

    X = citywide_df[[
        "Year", "Month", "Week", "Population",
        "average_weekly_temperature",
        "average_weekly_relative_humidity",
        "total_weekly_rainfall",
        "average_weekly_wind_speed",
        "average_weekly_wind_direction"
    ]]
    y = citywide_df["risk_code"]

    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', ohe, categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    class_counts = Counter(y)

    n_neighbors = 3
    if min(class_counts.values()) >= n_neighbors:
        smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
        X_balanced, y_balanced = smote.fit_resample(X_processed, y)
    else:
        print("Skipping SMOTE due to insufficient samples in one or more classes.")
        X_balanced, y_balanced = X_processed, y

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_balanced, y_balanced, test_size=0.3, stratify=y_balanced, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=citywide_df['risk_code'].nunique(),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )
    param_grid = {
        'max_depth':[3,5,7],
        'learning_rate':[0.05,0.1],
        'n_estimators':[100,200],
        'reg_alpha':[0,0.1]
    }
        
    grid = GridSearchCV(model, param_grid, scoring='balanced_accuracy', cv=3, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_xgb = grid.best_estimator_

    os.makedirs("forecast", exist_ok=True)
    joblib.dump(best_xgb, "forecast/model_citywide.joblib")
    joblib.dump(preprocessor, "forecast/preprocessor_citywide.joblib")

    print("Citywide model training completed and saved successfully.")

if __name__ == "__main__":
    main()