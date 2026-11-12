# Logistics Data Analysis - Week 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("logistics_sample_dataset.csv")
print(df.head())
print(df.info())
print("Missing values:", df.isna().sum().sum())
print(df.describe())
print(df.groupby("Transport_Mode").agg(
    shipments=("Shipment_ID","count"),
    avg_days=("Delivery_Time_days","mean"),
    avg_cost=("Transport_Cost_INR","mean"),
    delay_rate=("Delayed","mean")))
print(df.groupby("Region")["Delayed"].mean())
print(df[["Distance_km","Shipment_Volume","Delivery_Time_days",
          "Transport_Cost_INR","Delayed"]].corr())

# Save figures
import os
os.makedirs("charts", exist_ok=True)
df.Delivery_Time_days.hist(bins=25)
plt.title("Delivery Time Distribution"); plt.xlabel("Days"); plt.ylabel("Shipments")
plt.tight_layout(); plt.savefig("charts/delivery_hist.png", dpi=160); plt.close()
df.groupby("Transport_Mode").Delivery_Time_days.mean().plot(kind="bar")
plt.title("Average Delivery Time by Mode"); plt.ylabel("Days")
plt.tight_layout(); plt.savefig("charts/mode_time.png", dpi=160); plt.close()
plt.scatter(df.Distance_km, df.Transport_Cost_INR, alpha=.45, s=12)
plt.title("Distance vs Transport Cost"); plt.xlabel("Distance (km)"); plt.ylabel("INR")
plt.tight_layout(); plt.savefig("charts/distance_cost.png", dpi=160); plt.close()
df.groupby("Region").Delayed.mean().plot(kind="bar")
plt.title("Delay Rate by Region"); plt.ylabel("Delay rate")
plt.tight_layout(); plt.savefig("charts/region_delay.png", dpi=160); plt.close()
