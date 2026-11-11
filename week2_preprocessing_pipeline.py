"""
Yuva Intern - Week 2: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Author: Joel Maju
Description: End-to-End Python Data Hygiene Pipeline simulating multi-hub logistics telemetry
             (DataCo / Olist schema), performing automated quality profiling, hierarchical MAR/MCAR
             imputation, Tukey IQR Winsorization, Isolation Forest outlier filtering, and
             multi-method feature scaling (Yeo-Johnson, RobustScaler, MinMaxScaler).
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ==============================================================================
# MODULE 1: Logistics Data Collection Simulation & Automated Quality Profiler
# ==============================================================================
def simulate_and_profile_logistics_data(n_records: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates raw multi-hub logistics dataset with realistic missingness (MCAR/MAR),
    duplicates, and sensor outliers, then generates an automated Data Quality Audit report.
    """
    np.random.seed(seed)
    base_dates = pd.date_range("2026-08-01", periods=n_records, freq="15min")
    lead_hrs = np.random.gamma(shape=2.2, scale=18.0, size=n_records)

    df = pd.DataFrame({
        "shipment_id": [f"SHP-{100000 + i}" for i in range(n_records)],
        "hub_code": np.random.choice(["HUB-COK", "hub-cok ", "HUB_TCR", "HUB-KTM", "HUB-CCJ"], size=n_records),
        "product_category": np.random.choice(["Electronics", "FMCG", "Apparel", "ColdChain"], size=n_records),
        "order_timestamp": base_dates,
        "delivery_timestamp": base_dates + pd.to_timedelta(lead_hrs, unit="h"),
        "transit_lead_hours": lead_hrs.round(2),
        "package_weight_kg": np.random.lognormal(mean=1.4, sigma=0.65, size=n_records).round(2),
        "freight_cost_inr": np.random.lognormal(mean=4.8, sigma=0.55, size=n_records).round(2),
        "warehouse_stock_qty": np.random.randint(10, 2500, size=n_records).astype(float),
        "dest_lat": np.random.uniform(9.80, 10.25, size=n_records).round(6),
        "dest_lon": np.random.uniform(76.20, 76.50, size=n_records).round(6),
    })

    # Inject Realistic Anomalies: Missing Values (MAR/MCAR), Outliers & Sensor Errors
    df.loc[np.random.choice(df.index, 380, replace=False), "package_weight_kg"] = np.nan
    df.loc[np.random.choice(df.index, 290, replace=False), "transit_lead_hours"] = np.nan
    df.loc[np.random.choice(df.index, 180, replace=False), "freight_cost_inr"] = np.nan
    df.loc[np.random.choice(df.index, 45, replace=False), "package_weight_kg"] = 999.9   # Weighbridge spike
    df.loc[np.random.choice(df.index, 30, replace=False), "transit_lead_hours"] = -12.5  # Clock drift error
    df.loc[np.random.choice(df.index, 25, replace=False), ["dest_lat", "dest_lon"]] = 0.0  # Null Island GPS

    # Inject 100 duplicate webhook retries
    df_raw = pd.concat([df, df.sample(100, random_state=seed)], ignore_index=True)

    # Generate Automated Data Quality Audit Table
    num_skew = df_raw.select_dtypes(include=[np.number]).skew().round(2)
    audit_report = pd.DataFrame({
        "dtype": df_raw.dtypes.astype(str),
        "missing_count": df_raw.isna().sum(),
        "missing_pct": (df_raw.isna().mean() * 100).round(2),
        "unique_vals": df_raw.nunique(),
        "skewness": df_raw.columns.map(lambda c: num_skew.get(c, np.nan)),
    })
    return df_raw, audit_report


# ==============================================================================
# MODULE 2: Hierarchical Imputation, IQR Winsorization & Isolation Forest Hygiene
# ==============================================================================
def clean_and_treat_outliers(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Executes structural deduplication, string canonicalization, domain validation,
    hierarchical MAR imputation, IQR Winsorization, and Isolation Forest filtering.
    """
    df = df_raw.copy()

    # 1. Deterministic Deduplication & String Canonicalization
    df = df.drop_duplicates(subset=["shipment_id"], keep="last").reset_index(drop=True)
    df["hub_code"] = (
        df["hub_code"].astype(str).str.strip().str.upper().str.replace("_", "-", regex=False)
    )

    # 2. Domain Integrity Rules: Nullify negative lead times & Null-Island GPS pings
    df.loc[df["transit_lead_hours"] <= 0, "transit_lead_hours"] = np.nan
    df.loc[df["warehouse_stock_qty"] < 0, "warehouse_stock_qty"] = 0.0
    invalid_gps = ~df["dest_lat"].between(8.15, 12.85) | ~df["dest_lon"].between(74.80, 77.40)
    df.loc[invalid_gps, ["dest_lat", "dest_lon"]] = np.nan

    # 3. Hierarchical Group-Median Imputation for Cargo Weight & Geospatial Centroids
    df["package_weight_kg"] = df.groupby("product_category")["package_weight_kg"].transform(
        lambda x: x.fillna(x.median())
    )
    df["dest_lat"] = df.groupby("hub_code")["dest_lat"].transform(lambda x: x.fillna(x.median()))
    df["dest_lon"] = df.groupby("hub_code")["dest_lon"].transform(lambda x: x.fillna(x.median()))

    # 4. Multivariate KNN Imputation for Correlated Transit Lead Time & Freight Cost
    knn_cols = ["transit_lead_hours", "package_weight_kg", "freight_cost_inr", "dest_lat", "dest_lon"]
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    df[knn_cols] = imputer.fit_transform(df[knn_cols])

    # 5. Univariate Tukey IQR Fence Winsorization (Capping Extreme Tails)
    for col in ["transit_lead_hours", "package_weight_kg", "freight_cost_inr"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence, upper_fence = max(0.0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
        df[f"{col}_clean"] = df[col].clip(lower=lower_fence, upper=upper_fence)

    # 6. Multivariate Outlier Flagging via Isolation Forest
    iso_features = ["transit_lead_hours_clean", "package_weight_kg_clean", "freight_cost_inr_clean"]
    iso = IsolationForest(contamination=0.015, random_state=42)
    df["is_multivariate_outlier"] = (iso.fit_predict(df[iso_features]) == -1).astype(int)

    return df[df["is_multivariate_outlier"] == 0].reset_index(drop=True)


# ==============================================================================
# MODULE 3: Multi-Method Feature Normalization & Scikit-Learn ColumnTransformer
# ==============================================================================
def normalize_and_engineer_features(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, ColumnTransformer]:
    """
    Applies Yeo-Johnson power transformation to skewed lead times, RobustScaler to
    heavy-tailed freight/weight metrics, MinMaxScaler to inventory/coordinates,
    and OneHotEncoder to categorical logistics nodes.
    """
    df = df_clean.copy()

    # Cyclical Temporal Feature Encoding (Preserving 24-Hour Continuity)
    df["order_hour_sin"] = np.sin(2 * np.pi * df["order_timestamp"].dt.hour / 24.0)
    df["order_hour_cos"] = np.cos(2 * np.pi * df["order_timestamp"].dt.hour / 24.0)

    skewed_cols = ["transit_lead_hours_clean"]
    robust_cols = ["package_weight_kg_clean", "freight_cost_inr_clean"]
    minmax_cols = ["warehouse_stock_qty", "dest_lat", "dest_lon"]
    cat_cols = ["hub_code", "product_category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("yeo_johnson_skew", PowerTransformer(method="yeo-johnson", standardize=True), skewed_cols),
            ("robust_scaler", RobustScaler(quantile_range=(25.0, 75.0)), robust_cols),
            ("minmax_norm", MinMaxScaler(feature_range=(0.0, 1.0)), minmax_cols),
            ("cat_onehot", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ],
        remainder="passthrough",
    )

    transformed_array = preprocessor.fit_transform(
        df[skewed_cols + robust_cols + minmax_cols + cat_cols]
    )
    ohe_names = preprocessor.named_transformers_["cat_onehot"].get_feature_names_out(cat_cols).tolist()
    all_feature_names = (
        ["lead_time_yeojohnson", "weight_robust", "freight_robust",
         "stock_minmax", "lat_minmax", "lon_minmax"] + ohe_names
    )

    df_normalized = pd.DataFrame(transformed_array, columns=all_feature_names, index=df.index)
    df_normalized["order_hour_sin"] = df["order_hour_sin"].round(4)
    df_normalized["order_hour_cos"] = df["order_hour_cos"].round(4)

    return df_normalized, preprocessor


if __name__ == "__main__":
    print("=== OmniSwift Logistics: Week 2 Data Cleaning & Preprocessing Pipeline ===")
    raw_df, audit_df = simulate_and_profile_logistics_data(n_records=5000)
    print(f"\n[1] Raw Dataset Ingested: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(audit_df[["dtype", "missing_count", "missing_pct", "skewness"]].to_string())

    clean_df = clean_and_treat_outliers(raw_df)
    print(f"\n[2] Post-Cleaning & Outlier Hygiene Dataset: {clean_df.shape[0]} rows remaining")
    print(f"    Remaining Missing Values: {clean_df.isna().sum().sum()}")

    norm_df, pipeline_obj = normalize_and_engineer_features(clean_df)
    print(f"\n[3] Normalized & Feature-Engineered Matrix: {norm_df.shape[0]} rows, {norm_df.shape[1]} features")
    print(norm_df.head(5).round(3).to_string())
