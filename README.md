# CI/CD Failure Prediction & Deployment Risk Scorer

This project predicts CI/CD failure risk using GitHub Actions data, commit metadata, test reports, Azure Data Factory, ADLS Gen2, Databricks, PySpark, SparkSQL, Azure ML, MLflow, GitHub Actions, Azure DevOps, Terraform, and Streamlit.

## Architecture

1. Ingestion - Azure Data Factory pulls GitHub API data
2. Bronze - Raw JSON stored in ADLS Gen2
3. Silver - Databricks + PySpark clean and flatten data
4. Gold - Feature engineering for ML
5. ML - XGBoost model tracked with MLflow
6. Serving - Azure ML online endpoint
7. CI/CD - GitHub Actions + Azure DevOps
8. Dashboard - Streamlit app

## Build Order

1. Infrastructure
2. Ingestion
3. Bronze
4. Silver
5. Features
6. Model
7. Serving
8. CI/CD
9. Dashboard