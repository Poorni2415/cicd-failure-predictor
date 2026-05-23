# Databricks notebook source
# ── Full debug cell — run this before anything else ──
import mlflow

mlflow.set_registry_uri("databricks")
EXPERIMENT_NAME = "/Shared/cicd-failure-predictor"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print("❌ Experiment not found — check the name")
else:
    print(f"✅ Experiment found: {experiment.experiment_id}")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=5
    )
    
    print(f"\nTotal runs found: {len(runs)}")
    print(f"\nAll columns:")
    print(runs.columns.tolist())
    
    if not runs.empty:
        print(f"\nFirst run data:")
        print(runs.iloc[0])

# COMMAND ----------

# ============================================================
# CELL 0 — Config
# ============================================================
import mlflow

# Match whatever you used in Cell 5 of training notebook
mlflow.set_registry_uri("databricks")   

MODEL_NAME      = "cicd-failure-predictor"
MIN_AUC         = 0.55    # absolute minimum — below this is worse than random
EXPERIMENT_NAME = "/Shared/cicd-failure-predictor"

print("✅ Config loaded")

# COMMAND ----------

# DBTITLE 1,Cell 2
# ============================================================
# CELL 1 — Get latest FINISHED run AUC
# ============================================================
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id

# Only get FINISHED runs — ignore RUNNING/KILLED ones
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1
)

print(f"Completed runs found: {len(runs)}")

if runs.empty:
    raise Exception("No finished runs found")

latest_run = runs.iloc[0]
new_auc    = latest_run["metrics.auc"]
new_run_id = latest_run["run_id"]

print(f"Run ID: {new_run_id}")
print(f"AUC:    {new_auc:.4f}")

# COMMAND ----------

# ============================================================
# CELL 2 — Get current production model AUC (if exists)
# ============================================================
from mlflow.tracking import MlflowClient

client = MlflowClient()

try:
    # Get all versions of the model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    # Find the one tagged as Production
    prod_versions = [
        v for v in versions 
        if v.current_stage == "Production"
    ]
    
    if prod_versions:
        prod_version    = prod_versions[0]
        prod_run_id     = prod_version.run_id
        prod_run        = client.get_run(prod_run_id)
        prod_auc        = prod_run.data.metrics.get("auc", 0.0)
        print(f"Production model version: {prod_version.version}")
        print(f"Production AUC:           {prod_auc:.4f}")
    else:
        prod_auc = 0.0
        print("No production model exists yet — first deployment")
        print(f"Using baseline AUC: {prod_auc}")

except Exception as e:
    prod_auc = 0.0
    print(f"No production model found: {e}")
    print(f"Using baseline AUC: {prod_auc}")

# COMMAND ----------

# ============================================================
# CELL 3 — Gate decision
# ============================================================

print("\n" + "="*50)
print("MODEL EVALUATION GATE")
print("="*50)
print(f"New model AUC:        {new_auc:.4f}")
print(f"Production model AUC: {prod_auc:.4f}")
print(f"Minimum AUC:          {MIN_AUC:.4f}")
print("="*50)

beats_production = new_auc > prod_auc
beats_minimum    = new_auc >= MIN_AUC

if beats_production and beats_minimum:
    print("✅ GATE PASSED — new model is better")
    print("   Promoting to Production...")
    PROMOTE = True
elif not beats_minimum:
    print(f"❌ GATE FAILED — AUC {new_auc:.4f} below minimum {MIN_AUC:.4f}")
    PROMOTE = False
else:
    print(f"❌ GATE FAILED — new AUC {new_auc:.4f} does not beat production {prod_auc:.4f}")
    PROMOTE = False

# COMMAND ----------

# ============================================================
# CELL 4 — Promote if gate passed
# ============================================================

if PROMOTE:
    # Get the latest registered version
    versions     = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_ver   = sorted(versions, key=lambda v: int(v.version))[-1]
    
    # Archive current production model first
    prod_versions = [
        v for v in versions 
        if v.current_stage == "Production"
    ]
    for v in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived"
        )
        print(f"📦 Archived old production version {v.version}")
    
    # Promote new model to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_ver.version,
        stage="Production"
    )
    
    print(f"🚀 Version {latest_ver.version} promoted to Production")
    print(f"   AUC: {new_auc:.4f}")

else:
    print("🚫 Model rejected — production model unchanged")
    raise Exception(
        f"Evaluation gate failed — "
        f"new AUC {new_auc:.4f} did not beat production {prod_auc:.4f}"
    )

# COMMAND ----------

# ============================================================
# CELL 5 — Log gate result back to the run
# ============================================================
with mlflow.start_run(run_id=new_run_id):
    mlflow.set_tag("gate_result",    "PASSED" if PROMOTE else "FAILED")
    mlflow.set_tag("promoted",       str(PROMOTE))
    mlflow.set_tag("prod_auc_at_time", str(round(prod_auc, 4)))

print("✅ Gate result logged to MLflow run")
print(f"\nSummary:")
print(f"  Run ID:    {new_run_id}")
print(f"  New AUC:   {new_auc:.4f}")
print(f"  Prod AUC:  {prod_auc:.4f}")
print(f"  Decision:  {'PROMOTED ✅' if PROMOTE else 'REJECTED ❌'}")