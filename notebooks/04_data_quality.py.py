# Databricks notebook source
# ============================================================
# CELL 0 — Reinstall correct version
# ============================================================
%pip install great-expectations==0.17.23
dbutils.library.restartPython()

# COMMAND ----------

import great_expectations as gx
print(gx.__version__)  # should print 0.17.23

# COMMAND ----------

# ============================================================
# CELL 1 — Config
# ============================================================
STORAGE_ACCOUNT = "cicdriskpoornima001"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    "1u/MPlueAq8m+JqUd2qEziCI5t2D+tHUgR+bWYPKsXfd0/FMWq3hRV0IKINyPsGpFZRJz4ktb2y8+AStYFJvkA=="
)

SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"

# Load silver_workflow_runs as pandas for Great Expectations
df_runs_spark = spark.read.format("delta").load(
    f"{SILVER_BASE}/delta/silver_workflow_runs"
)

# Convert to pandas (GE works on pandas dataframes)
df_runs = df_runs_spark.toPandas()

print(f"Rows loaded: {len(df_runs)}")
print(df_runs.dtypes)

# COMMAND ----------

# Run this first to check version
import great_expectations as gx
print(gx.__version__)

# COMMAND ----------

# ============================================================
# CELL 2 — GE context (0.17.x compatible)
# ============================================================
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    InMemoryStoreBackendDefaults
)

# In-memory context — no filesystem needed
config = DataContextConfig(
    store_backend_defaults=InMemoryStoreBackendDefaults()
)
context = BaseDataContext(project_config=config)

# Add pandas datasource
context.add_datasource(
    name="workflow_runs_source",
    class_name="Datasource",
    execution_engine={"class_name": "PandasExecutionEngine"},
    data_connectors={
        "runtime_connector": {
            "class_name": "RuntimeDataConnector",
            "batch_identifiers": ["batch_id"]
        }
    }
)

batch_request = RuntimeBatchRequest(
    datasource_name="workflow_runs_source",
    data_connector_name="runtime_connector",
    data_asset_name="workflow_runs",
    runtime_parameters={"batch_data": df_runs},
    batch_identifiers={"batch_id": "phase6_batch"}
)

# Create expectation suite
context.create_expectation_suite(
    "silver_workflow_runs_suite", 
    overwrite_existing=True
)

validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="silver_workflow_runs_suite"
)

print(f"✅ GE context ready — GE version: {gx.__version__}")

# COMMAND ----------

# ============================================================
# CELL 3 — Define expectations (0.17.x style)
# ============================================================

validator.expect_column_values_to_not_be_null("run_id")
validator.expect_column_values_to_not_be_null("repo")
validator.expect_column_values_to_not_be_null("created_at")

validator.expect_column_values_to_be_in_set(
    "is_failed", [0, 1]
)

validator.expect_column_values_to_be_between(
    "duration_mins", min_value=0, max_value=500
)

validator.expect_column_values_to_be_in_set(
    "conclusion",
    ["success", "failure", "cancelled", "skipped", "timed_out", "neutral"]
)

validator.save_expectation_suite(discard_failed_expectations=False)
print("✅ Expectations defined")

# COMMAND ----------

# ============================================================
# CELL 4 — Run validation (0.17.x style)
# ============================================================

results = validator.validate()

print("\n" + "="*50)
print("DATA QUALITY REPORT — silver_workflow_runs")
print("="*50)

passed = 0
failed = 0

for result in results.results:
    status  = "✅ PASS" if result.success else "❌ FAIL"
    col     = result.expectation_config.kwargs.get("column", "N/A")
    rule    = result.expectation_config.expectation_type
    print(f"{status} | {col:20s} | {rule}")
    if result.success:
        passed += 1
    else:
        failed += 1

print("="*50)
print(f"Total: {passed} passed, {failed} failed")
print(f"Overall: {'✅ PASSED' if failed == 0 else '❌ FAILED'}")

# COMMAND ----------

# ============================================================
# CELL 5 — Gate: stop pipeline if critical checks fail
# ============================================================

# Critical columns — if these fail, something is seriously wrong
critical_columns = ["run_id", "repo", "created_at"]

critical_failures = [
    r for r in results.results
    if not r.success
    and r.expectation_config.kwargs.get("column") in critical_columns
]

if critical_failures:
    print("🚨 CRITICAL DATA QUALITY FAILURE — stopping pipeline")
    for f in critical_failures:
        print(f"  ❌ {f.expectation_config.kwargs.get('column')} failed")
    raise Exception("Data quality gate failed — Silver data has critical nulls")
else:
    print("✅ Critical checks passed — safe to proceed to Gold layer")
    print("   (Non-critical failures logged but pipeline continues)")

# COMMAND ----------

# ============================================================
# CELL 6 — Validate silver_commits (0.17.23 style)
# ============================================================
df_commits = spark.read.format("delta").load(
    f"{SILVER_BASE}/delta/silver_commits"
).toPandas()

print(f"Commits rows loaded: {len(df_commits)}")

# Add new datasource for commits
context.add_datasource(
    name="commits_source",
    class_name="Datasource",
    execution_engine={"class_name": "PandasExecutionEngine"},
    data_connectors={
        "runtime_connector": {
            "class_name": "RuntimeDataConnector",
            "batch_identifiers": ["batch_id"]
        }
    }
)

from great_expectations.core.batch import RuntimeBatchRequest

batch_request2 = RuntimeBatchRequest(
    datasource_name="commits_source",
    data_connector_name="runtime_connector",
    data_asset_name="commits",
    runtime_parameters={"batch_data": df_commits},
    batch_identifiers={"batch_id": "commits_batch"}
)

context.create_expectation_suite(
    "silver_commits_suite",
    overwrite_existing=True
)

validator2 = context.get_validator(
    batch_request=batch_request2,
    expectation_suite_name="silver_commits_suite"
)

# ── Expectations ──
validator2.expect_column_values_to_not_be_null("commit_sha")
validator2.expect_column_values_to_not_be_null("repo")

# additions/deletions may be null (GitHub list API doesn't return stats)
# so we only check non-null rows are >= 0
validator2.expect_column_values_to_be_between(
    "additions", min_value=0, mostly=0.5   # at least 50% of non-null rows
)
validator2.expect_column_values_to_be_between(
    "deletions", min_value=0, mostly=0.5
)

validator2.save_expectation_suite(discard_failed_expectations=False)

results2 = validator2.validate()

print("\n=== DATA QUALITY REPORT — silver_commits ===")
passed = failed = 0
for result in results2.results:
    status = "✅ PASS" if result.success else "❌ FAIL"
    col    = result.expectation_config.kwargs.get("column", "N/A")
    rule   = result.expectation_config.expectation_type
    print(f"{status} | {col:20s} | {rule}")
    if result.success: passed += 1
    else: failed += 1

print(f"\nTotal: {passed} passed, {failed} failed")