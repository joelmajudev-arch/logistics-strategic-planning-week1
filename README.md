# Strategic Planning and Data Exploration in Logistics (Week 1)

**Author:** Joel Maju  
**Program:** Yuva Intern — Data Science & Supply Chain Analytics  
**Deliverable:** Week 1 Strategic Planning Report & Python Analytics Architecture

---

## 📌 Project Overview
This repository contains the strategic planning framework and Python reference implementation for **OmniSwift Regional Fulfillment Network**, addressing three high-cost supply chain bottlenecks:
1. **Inventory Stockout-Overstock Imbalance:** Dynamic safety stock and Reorder Point (ROP) calculation coupled with LightGBM demand forecasting.
2. **Static Delivery Zone Fragmentation:** Spatial micro-fulfillment clustering using HDBSCAN and volume-weighted K-Means.
3. **Sub-Optimal Last-Mile Fleet Routing:** Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) optimization using Google OR-Tools and spatial distance matrices.

---

## 📊 Key Performance Indicators (KPIs)
| KPI Metric | Category | Current Baseline | Target Benchmark |
| :--- | :--- | :---: | :---: |
| **On-Time In-Full (OTIF)** | Service Reliability | 84.2% | >= 96.5% |
| **Forecast Error (WMAPE)** | Inventory Planning | 28.4% | <= 11.5% |
| **Inventory Turnover Ratio (ITR)** | Asset Efficiency | 6.4x / yr | 10.2x / yr |
| **Cost Per Order Delivered (CPOD)** | Fleet Economics | ₹ 68.50 | ₹ 52.00 |
| **Fleet Capacity Utilization (FCU)** | Resource Allocation | 63.8% | >= 86.0% |

---

## 🚀 Repository Structure
```text
├── logistics_pipeline.py                        # End-to-end Python analytics & optimization script
├── requirements.txt                             # Required Python libraries
├── Week1_Strategic_Planning_Logistics_Report.docx # Full Strategic Planning Report (DOCX)
└── README.md                                    # Project documentation
```

---

## 🛠️ Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the diagnostic and optimization simulation pipeline:
   ```bash
   python logistics_pipeline.py
   ```
