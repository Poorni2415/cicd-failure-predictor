# Databricks notebook source
# ============================================================
# CELL 0 — Reinstall with pinned marshmallow (fixes _T error)
# ============================================================
%pip install marshmallow==3.19.0
%pip install azure-ai-ml==1.12.0 azure-identity==1.15.0 xgboost mlflow
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks")

MODEL_NAME    = "cicd-failure-predictor"
MODEL_VERSION = 2        # the version that got promoted to Production

print("✅ Config ready")

# COMMAND ----------

# ============================================================
# CELL 2 — Load and test model locally (fixed)
# ============================================================
import mlflow
import mlflow.xgboost
import pandas as pd

mlflow.set_registry_uri("databricks")

MODEL_NAME    = "cicd-failure-predictor"
MODEL_VERSION = 2   # your promoted version number

# ── Correct URI format for legacy Workspace Registry ──
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"  # use version number not stage

loaded_model = mlflow.xgboost.load_model(model_uri)
print("✅ Model loaded from registry")

# ── Sanity test ──
FEATURE_COLS = [
    "duration_mins", "run_attempt", "is_friday",
    "rolling_failure_rate", "rolling_build_count",
    "files_changed",          # ← add this back
    "author_failure_rate", "author_total_builds",
    "is_draft", "is_merged", "pr_age_hours",
    "is_main_branch", "daily_build_count",
    "daily_failure_rate", "avg_duration_mins",
]

test_input = pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}])
test_input["rolling_failure_rate"] = 0.35
test_input["is_friday"]            = 1
test_input["duration_mins"]        = 5.0

risk_score = loaded_model.predict_proba(test_input)[:, 1][0]
risk_level = "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"

print(f"✅ Local inference works")
print(f"   Risk score: {risk_score:.4f}")
print(f"   Risk level: {risk_level}")

# COMMAND ----------

# ── Write scoring script to disk ──
# This is what the endpoint will run for every request

scoring_script = '''
import mlflow
import pandas as pd
import json

FEATURE_COLS = [
    "duration_mins", "run_attempt", "is_friday",
    "rolling_failure_rate", "rolling_build_count",
    "files_changed",          # ← add here too
    "author_failure_rate", "author_total_builds",
    "is_draft", "is_merged", "pr_age_hours",
    "is_main_branch", "daily_build_count",
    "daily_failure_rate", "avg_duration_mins",
]

def init():
    global model
    import os
    model_path = os.getenv("AZUREML_MODEL_DIR", ".")
    model = mlflow.xgboost.load_model(model_path)
    print("Model loaded")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df   = pd.DataFrame([data])
        df   = df.reindex(columns=FEATURE_COLS, fill_value=0)
        
        risk_score = float(model.predict_proba(df)[:, 1][0])
        risk_level = (
            "high"   if risk_score > 0.7 else
            "medium" if risk_score > 0.4 else
            "low"
        )
        
        return json.dumps({
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
'''

with open("/tmp/score.py", "w") as f:
    f.write(scoring_script)

print("✅ Scoring script written to /tmp/score.py")

# COMMAND ----------

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

# ── Connect to Azure ML ──
credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id="5ce8c57f-d22b-4667-92f0-10fca79c6871",    # Azure Portal → Subscriptions
    resource_group_name="rg-cicdrisk",
    workspace_name="mlw-cicdrisk-poornima"   # your Azure ML workspace name
)

print("✅ Connected to Azure ML")

# ── Create endpoint ──
endpoint_name = "cicd-risk-endpoint"

endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="CI/CD failure risk scoring endpoint",
    auth_mode="key"
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"✅ Endpoint created: {endpoint_name}")

# COMMAND ----------

# ── Deploy model to endpoint ──
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=f"azureml://registries/cicd-failure-predictor/models/{MODEL_NAME}/versions/{MODEL_VERSION}",
    environment=Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file={
            "dependencies": [
                "python=3.9",
                {"pip": ["xgboost", "mlflow", "pandas", "scikit-learn"]}
            ]
        }
    ),
    code_configuration=CodeConfiguration(
        code="/tmp",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(deployment).result()
print("✅ Deployment complete")

# ── Route 100% traffic to blue deployment ──
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("✅ Traffic routed to blue deployment")

# COMMAND ----------

# ── Get endpoint URL and key ──
endpoint_details = ml_client.online_endpoints.get(endpoint_name)
scoring_uri      = endpoint_details.scoring_uri

keys = ml_client.online_endpoints.get_keys(endpoint_name)
api_key = keys.primary_key

print(f"Endpoint URL: {scoring_uri}")
print(f"API Key:      {api_key[:10]}...")

# ── Send a test request ──
import requests
import json

test_payload = {
    "duration_mins":        8.5,
    "run_attempt":          2,
    "is_friday":            1,
    "rolling_failure_rate": 0.42,
    "rolling_build_count":  15,
    "files_changed":        8,    # ← add this
    "author_failure_rate":  0.31,
    "author_total_builds":  30,
    "is_draft":             0,
    "is_merged":            0,
    "pr_age_hours":         24.0,
    "is_main_branch":       1,
    "daily_build_count":    20,
    "daily_failure_rate":   0.35,
    "avg_duration_mins":    6.0,
}

response = requests.post(
    scoring_uri,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    },
    data=json.dumps(test_payload)
)

print(f"\nStatus: {response.status_code}")
print(f"Response: {response.json()}")