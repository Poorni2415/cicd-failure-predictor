# Databricks notebook source
# ============================================================
# CELL 0 — Install
# ============================================================
%pip install xgboost mlflow azureml-mlflow scikit-learn
dbutils.library.restartPython()

# COMMAND ----------

# ============================================================
# CELL 1 — Config
# ============================================================
STORAGE_ACCOUNT = "cicdriskpoornima001"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    "1u/MPlueAq8m+JqUd2qEziCI5t2D+tHUgR+bWYPKsXfd0/FMWq3hRV0IKINyPsGpFZRJz4ktb2y8+AStYFJvkA=="
)

GOLD_BASE = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print("✅ Config loaded")

# COMMAND ----------

# ============================================================
# CELL 2 — Load gold table
# ============================================================
df_gold = spark.read.format("delta").load(
    f"{GOLD_BASE}/delta/gold_ci_failure_features"
)

print(f"Total rows: {df_gold.count()}")
print(f"Failed:  {df_gold.filter('label=1').count()}")
print(f"Success: {df_gold.filter('label=0').count()}")

# Convert to pandas for XGBoost
pdf = df_gold.toPandas()
print(f"\nColumns: {list(pdf.columns)}")

# COMMAND ----------

# ============================================================
# CELL 3 — Feature preparation (fixed)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Check exactly what columns exist in your gold table ──
print("Available columns:")
print(pdf.columns.tolist())

# ── Use only columns that actually exist ──
ALL_POSSIBLE_FEATURES = [
    "duration_mins",
    "run_attempt",
    "is_friday",
    "rolling_failure_rate",
    "rolling_build_count",
    "additions",
    "deletions",
    "files_changed",
    "author_failure_rate",
    "author_total_builds",
    "is_draft",
    "is_merged",
    "pr_age_hours",
    "is_main_branch",
    "daily_build_count",
    "daily_failure_rate",
    "avg_duration_mins",
]

# ── Only keep features that are actually present ──
FEATURE_COLS = [c for c in ALL_POSSIBLE_FEATURES if c in pdf.columns]

print(f"\nFeatures selected ({len(FEATURE_COLS)}):")
print(FEATURE_COLS)

TARGET = "label"

# ── Fill nulls ──
pdf[FEATURE_COLS] = pdf[FEATURE_COLS].fillna(0)

X = pdf[FEATURE_COLS]
y = pdf[TARGET]

# ── Train/test split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain: {len(X_train)} rows")
print(f"Test:  {len(X_test)} rows")
print(f"Failure rate in train: {y_train.mean():.2%}")
print(f"Failure rate in test:  {y_test.mean():.2%}")

# COMMAND ----------

# ============================================================
# CELL 4 — Train XGBoost + MLflow (fixed for small datasets)
# ============================================================
mlflow.end_run()

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use('Agg')          # ← fixes matplotlib path error on Databricks
import matplotlib.pyplot as plt

# ── Check label distribution before training ──
print(f"Label distribution in full dataset:")
print(y.value_counts())
print(f"Total failures: {(y==1).sum()}")
print(f"Total successes: {(y==0).sum()}")

if (y==1).sum() == 0:
    print("\n⚠️ No failed builds in dataset yet.")
    print("This is normal with 1 day of data from a healthy repo.")
    print("Using synthetic minority to demonstrate the pipeline...")
    
    # ── Inject a few synthetic failures so pipeline can run ──
    import numpy as np
    n_synthetic = max(5, int(len(X) * 0.1))  # 10% synthetic failures
    synthetic_X = X.sample(n=n_synthetic, random_state=42).copy()
    synthetic_y = pd.Series([1] * n_synthetic)
    
    X_aug = pd.concat([X, synthetic_X], ignore_index=True)
    y_aug = pd.concat([y, synthetic_y], ignore_index=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug,
        test_size=0.2,
        random_state=42,
        stratify=y_aug
    )
    print(f"Augmented — Train: {len(X_train)}, Test: {len(X_test)}")
else:
    print("✅ Real failures found — training on actual data")

# ── Safe scale_pos_weight ──
n_neg = len(y_train[y_train==0])
n_pos = len(y_train[y_train==1])
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

mlflow.set_experiment("/Shared/cicd-failure-predictor")

params = {
    "n_estimators":      200,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  scale_pos_weight,
    "random_state":      42,
    "eval_metric":       "auc",
}

with mlflow.start_run(run_name="xgboost_baseline") as run:

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = model.predict(X_test)

    auc       = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*40}")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"{'='*40}")

    mlflow.log_params(params)
    mlflow.log_metric("auc",       auc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1",        f1)

    # ── Confusion matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay(
        cm, display_labels=["success","failure"]
    ).plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)      # ← close instead of show (fixes Databricks display issue)

    # ── Feature importance ──
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(model, ax=ax2, max_num_features=15)
    plt.title("Feature Importance")
    plt.tight_layout()
    mlflow.log_figure(fig2, "feature_importance.png")
    plt.close(fig2)

    mlflow.xgboost.log_model(model, "xgboost_model")
    run_id = run.info.run_id
    print(f"\n✅ MLflow run logged: {run_id}")

# COMMAND ----------

# ============================================================
# CELL 5 — Register model (Unity Catalog format)
# ============================================================

# Option A — Use legacy Workspace Registry (simplest fix, works immediately)
mlflow.set_registry_uri("databricks")

model_uri  = f"runs:/{run_id}/xgboost_model"
model_name = "cicd-failure-predictor"

registered = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f"✅ Model registered")
print(f"   Name:    {registered.name}")
print(f"   Version: {registered.version}")

# COMMAND ----------

# ============================================================
# CELL 6 — Save feature metadata
# (needed by the scoring endpoint in Phase 10)
# ============================================================
import json

feature_metadata = {
    "feature_cols":   FEATURE_COLS,
    "target":         TARGET,
    "model_name":     model_name,
    "run_id":         run_id,
    "auc":            round(auc, 4),
    "train_rows":     len(X_train),
    "test_rows":      len(X_test),
}

with open("/tmp/feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=2)

mlflow.log_artifact("/tmp/feature_metadata.json")

print("✅ Feature metadata saved")
print(json.dumps(feature_metadata, indent=2))

# COMMAND ----------

# ============================================================
# CELL 7 — Print link to MLflow experiment
# ============================================================
experiment = mlflow.get_experiment_by_name(
    "/Shared/cicd-failure-predictor"
)

print(f"View your experiment in MLflow UI:")
print(f"Databricks → Experiments → cicd-failure-predictor")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Run ID:        {run_id}")
print(f"AUC:           {auc:.4f}")

# COMMAND ----------

# How many rows did ingestion actually fetch?
STORAGE_ACCOUNT = "cicdriskpoornima001"
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    "1u/MPlueAq8m+JqUd2qEziCI5t2D+tHUgR+bWYPKsXfd0/FMWq3hRV0IKINyPsGpFZRJz4ktb2y8+AStYFJvkA=="
)
BRONZE_BASE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"

df_bronze_runs = spark.read.format("delta").load(
    f"{BRONZE_BASE}/delta/bronze_workflow_runs"
)
print(f"Bronze runs rows: {df_bronze_runs.count()}")
df_bronze_runs.select("conclusion").groupBy("conclusion").count().show()

# COMMAND ----------

SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"

df_silver_runs = spark.read.format("delta").load(
    f"{SILVER_BASE}/delta/silver_workflow_runs"
)
print(f"Silver runs rows: {df_silver_runs.count()}")
df_silver_runs.select("conclusion", "is_failed").groupBy("conclusion", "is_failed").count().show()

# COMMAND ----------

GOLD_BASE = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

df_gold_check = spark.read.format("delta").load(
    f"{GOLD_BASE}/delta/gold_ci_failure_features"
)
print(f"Gold rows: {df_gold_check.count()}")
df_gold_check.select("label").groupBy("label").count().show()

# COMMAND ----------

# Check raw JSON before Delta
runs_json = spark.read.option("multiLine", False).json(
    f"{BRONZE_BASE}/github_runs/repo=fastapi_fastapi/year=2026/month=05/day=02/runs.json"
)
print(f"Raw JSON rows: {runs_json.count()}")

# COMMAND ----------

