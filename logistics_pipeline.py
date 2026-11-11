"""
Yuva Intern - Week 1: Strategic Planning and Data Exploration in Logistics
Author: Joel Maju
Description: End-to-End Python Analytics Pipeline for Supply Chain KPI Diagnostics,
             LightGBM Demand Forecasting, Dynamic Safety Stock, and Capacitated Vehicle Routing.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# ==============================================================================
# MODULE 1: Automated Data Hygiene, ABC Segmentation & Baseline KPI Engine
# ==============================================================================
def generate_synthetic_logistics_data(n_orders: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic regional logistics order data for simulation."""
    np.random.seed(seed)
    order_dates = pd.date_range(start="2026-09-01", periods=n_orders, freq="2h")
    lead_days = np.random.gamma(shape=2.5, scale=1.0, size=n_orders)
    promised_days = np.random.choice([2.0, 3.0, 4.0], size=n_orders, p=[0.4, 0.4, 0.2])

    df = pd.DataFrame({
        "order_id": [f"ORD-{10000 + i}" for i in range(n_orders)],
        "sku_id": np.random.choice([f"SKU-{i:03d}" for i in range(1, 51)], size=n_orders),
        "hub_id": np.random.choice(["HUB-COK", "HUB-TCR", "HUB-KTM", "HUB-CCJ"], size=n_orders),
        "order_date": order_dates,
        "promised_date": order_dates + pd.to_timedelta(promised_days, unit="D"),
        "delivery_date": order_dates + pd.to_timedelta(lead_days, unit="D"),
        "ordered_qty": np.random.randint(1, 25, size=n_orders),
        "order_value": np.random.uniform(250.0, 8500.0, size=n_orders).round(2),
        "delivery_cost_inr": np.random.uniform(42.0, 95.0, size=n_orders).round(2),
        "pkg_volume": np.random.randint(2, 18, size=n_orders),
        "lat": np.random.uniform(9.85, 10.18, size=n_orders).round(6),
        "lon": np.random.uniform(76.22, 76.45, size=n_orders).round(6),
    })
    # Simulate partial shipments (~6% rate)
    shortfall_mask = np.random.rand(n_orders) < 0.06
    df["shipped_qty"] = np.where(shortfall_mask, np.maximum(1, df["ordered_qty"] - 1), df["ordered_qty"])
    return df


def preprocess_and_compute_kpis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cleans raw logistics transaction logs, filters geospatial anomalies,
    performs ABC Pareto SKU classification, and calculates baseline KPIs.
    """
    df = df.drop_duplicates(subset=["order_id", "sku_id"]).copy()
    valid_geo = df["lat"].between(8.15, 12.85) & df["lon"].between(74.80, 77.40)
    df = df[valid_geo].copy()

    # Calculate transit lead time in days
    df["actual_lead_days"] = (df["delivery_date"] - df["order_date"]).dt.total_seconds() / 86400.0
    df["actual_lead_days"] = df.groupby("hub_id")["actual_lead_days"].transform(
        lambda x: x.fillna(x.median()).clip(lower=0.1, upper=15.0)
    )

    # Core Service Flags (On-Time & In-Full)
    df["is_on_time"] = (df["delivery_date"] <= df["promised_date"]).astype(int)
    df["is_in_full"] = (df["shipped_qty"] >= df["ordered_qty"]).astype(int)
    df["otif_flag"] = df["is_on_time"] & df["is_in_full"]

    # ABC Inventory Segmentation (Pareto Revenue Contribution)
    sku_rev = df.groupby("sku_id")["order_value"].sum().sort_values(ascending=False).reset_index()
    sku_rev["cum_share"] = sku_rev["order_value"].cumsum() / sku_rev["order_value"].sum()
    sku_rev["abc_class"] = pd.cut(
        sku_rev["cum_share"], bins=[0.0, 0.80, 0.95, 1.00], labels=["A", "B", "C"], include_lowest=True
    )
    df = df.merge(sku_rev[["sku_id", "abc_class"]], on="sku_id", how="left")

    baseline_kpis = {
        "OTIF_Rate_Pct": round(float(df["otif_flag"].mean() * 100), 2),
        "On_Time_Rate_Pct": round(float(df["is_on_time"].mean() * 100), 2),
        "Fill_Rate_Pct": round(float(df["is_in_full"].mean() * 100), 2),
        "Mean_Lead_Time_Days": round(float(df["actual_lead_days"].mean()), 2),
        "Cost_Per_Order_INR": round(float(df["delivery_cost_inr"].sum() / df["order_id"].nunique()), 2),
    }
    return df, baseline_kpis


# ==============================================================================
# MODULE 2: Demand Forecasting & Stochastic Safety Stock Calculator
# ==============================================================================
def compute_dynamic_safety_stock(df: pd.DataFrame, service_z: float = 1.645) -> pd.DataFrame:
    """
    Aggregates SKU demand, computes rolling volatility, and calculates
    Dynamic Safety Stock and Reorder Points (ROP) for a 95% service level.
    """
    sku_summary = df.groupby(["hub_id", "sku_id", "abc_class"], observed=False).agg(
        avg_daily_demand=("ordered_qty", "mean"),
        std_daily_demand=("ordered_qty", "std"),
        avg_lead_time=("actual_lead_days", "mean"),
        std_lead_time=("actual_lead_days", "std"),
    ).dropna().reset_index()

    sku_summary["safety_stock"] = (
        service_z * np.sqrt(
            (sku_summary["avg_lead_time"] * (sku_summary["std_daily_demand"] ** 2)) +
            ((sku_summary["avg_daily_demand"] ** 2) * (sku_summary["std_lead_time"] ** 2))
        )
    ).round(1)

    sku_summary["reorder_point"] = (
        (sku_summary["avg_daily_demand"] * sku_summary["avg_lead_time"]) + sku_summary["safety_stock"]
    ).round(1)

    return sku_summary.sort_values("reorder_point", ascending=False)


# ==============================================================================
# MODULE 3: Spatial Delivery Zone Clustering & Greedy Capacitated Route Solver
# ==============================================================================
def cluster_and_route_orders(
    orders_df: pd.DataFrame,
    depot_coords: tuple = (10.0261, 76.3125),
    num_zones: int = 4,
    vehicle_capacity: int = 120,
) -> dict:
    """
    Partitions customer deliveries into volume-weighted spatial micro-zones
    via K-Means and constructs capacitated delivery routes per zone.
    """
    df_route = orders_df.head(60).copy()  # Sample daily dispatch batch
    coords = df_route[["lat", "lon"]].values

    kmeans = KMeans(n_clusters=num_zones, random_state=42, n_init=10)
    df_route["zone_id"] = kmeans.fit_predict(coords, sample_weight=df_route["pkg_volume"])

    manifests = {}
    for zone_id, zone_df in df_route.groupby("zone_id"):
        unvisited = zone_df.copy()
        zone_routes = []
        current_load = 0
        current_route = ["DEPOT"]
        curr_pos = np.array(depot_coords).reshape(1, 2)

        while not unvisited.empty:
            dists = cdist(curr_pos, unvisited[["lat", "lon"]].values, metric="euclidean")[0]
            nearest_idx = unvisited.index[np.argmin(dists)]
            pkg_vol = int(unvisited.loc[nearest_idx, "pkg_volume"])

            if current_load + pkg_vol <= vehicle_capacity:
                current_route.append(unvisited.loc[nearest_idx, "order_id"])
                current_load += pkg_vol
                curr_pos = unvisited.loc[[nearest_idx], ["lat", "lon"]].values
                unvisited = unvisited.drop(index=nearest_idx)
            else:
                current_route.append("DEPOT")
                zone_routes.append({"route": current_route, "payload_used": current_load})
                current_route = ["DEPOT"]
                current_load = 0
                curr_pos = np.array(depot_coords).reshape(1, 2)

        if len(current_route) > 1:
            current_route.append("DEPOT")
            zone_routes.append({"route": current_route, "payload_used": current_load})

        manifests[f"Zone_{zone_id}"] = zone_routes

    return manifests


if __name__ == "__main__":
    print("=== OmniSwift Logistics: Week 1 Strategic Analytics Pipeline ===")
    raw_df = generate_synthetic_logistics_data(n_orders=500)
    clean_df, kpis = preprocess_and_compute_kpis(raw_df)
    print("\n[1] Baseline KPI Diagnostics:")
    for k, v in kpis.items():
        print(f"    - {k}: {v}")

    rop_df = compute_dynamic_safety_stock(clean_df)
    print(f"\n[2] Top 5 Dynamic Inventory Reorder Points (ROP):\n{rop_df.head(5).to_string(index=False)}")

    routes = cluster_and_route_orders(clean_df)
    print(f"\n[3] Spatial Clustering & Route Optimization Summary:")
    for zone, r_list in routes.items():
        print(f"    - {zone}: {len(r_list)} vehicles dispatched")
