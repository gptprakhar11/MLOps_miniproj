# SCSE3040 MLOps Mini Project --- Group G111

## Problem Information
- **Problem ID:** MP266
- **Problem Title:** Cold Storage Spoilage (Surat)
- **Client:** Surat Cold Chain Cooperative
- **Batch:** 11
- **Faculty:** Pulkit Dwivedi

## Group Members
| Enrolment No. | Name | GitHub Username / Role |
|---|---|---|
| S24CSEU1001 | Aditya Raj | Data Lead |
| S24CSEU1002 | Aman Bhandari | Model Lead |
| S24CSEU1033 | Aayushman Mathpati | @Aayushman1119988 / Service Lead |
| S24CSEU1006 | Prakhar Gupta | Ops Lead |

## Project Overview
This repository contains the complete end-to-end MLOps pipeline for predicting cold storage spoilage percentage (`spoilage_pct`) of agricultural commodities (potatoes, onions, and apples) in cold storage chambers near Surat.

### Pipeline Stages
- **Week 1 / Stage 1:** Data Version Control (DVC) setup, exploratory data analysis, detection of 7 data quality issues in `train.csv`, and documentation in `findings.yaml`.
- **Stage 2:** Data cleaning pipeline, feature engineering, and baseline model training tracked with MLflow.
- **Stage 3:** Containerized FastAPI prediction service (`POST /predict`, `GET /health`), automated CI testing via GitHub Actions, and Kubernetes deployment configuration.
- **Stage 4:** Production monitoring (drift via PSI, group-wise MAE), automated model retraining/promotion, and holdout predictions.

### Week 1 Service Lead Deliverables
- Verified exploratory data analysis and 7 data quality checks.
- Documented data issues findings in `findings.yaml` and `reports/findings.yaml`.
