# Databricks notebook source
# ============================================================
# CELL 1 — Config
# ============================================================
STORAGE_ACCOUNT = "cicdriskpoornima001"   # e.g. adlscicdrisk
CONTAINER_BASE  = f"abfss://{{container}}@{STORAGE_ACCOUNT}.dfs.core.windows.net"

BRONZE_BASE = CONTAINER_BASE.format(container="bronze")
SILVER_BASE = CONTAINER_BASE.format(container="silver")  # for later

# Date partition (matches what Phase 3 wrote)
from datetime import date
today = date.today()
YEAR  = today.year
MONTH = str(today.month).zfill(2)
DAY   = str(today.day).zfill(2)

REPO_PARTITION = "repo=fastapi_fastapi"
DATE_PATH      = f"year={YEAR}/month={MONTH}/day={DAY}"

print(f"Processing partition: {REPO_PARTITION}/{DATE_PATH}")

# COMMAND ----------

# DBTITLE 1,Cell 2
# ============================================================
# CELL 2 — Bronze: workflow_runs
# ============================================================
from pyspark.sql.functions import lit, current_timestamp



STORAGE_ACCOUNT = "cicdriskpoornima001"

storage_account_key = "1u/MPlueAq8m+JqUd2qEziCI5t2D+tHUgR+bWYPKsXfd0/FMWq3hRV0IKINyPsGpFZRJz4ktb2y8+AStYFJvkA=="

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_account_key
)

runs_path = f"{BRONZE_BASE}/github_runs/{REPO_PARTITION}/{DATE_PATH}/runs.json"

df_runs = (
    spark.read
         .option("multiLine", False)
         .json(runs_path)
)

print(f"Rows read: {df_runs.count()}")
df_runs.printSchema()

# Add ingestion metadata columns
df_runs = (
    df_runs
    .withColumn("ingested_at", current_timestamp())
    .withColumn("source_repo", lit("fastapi/fastapi"))
    .withColumn("partition_date", lit(f"{YEAR}-{MONTH}-{DAY}"))
)

# Write as Delta — mergeSchema handles API schema drift over time
(
    df_runs.write
           .format("delta")
           .mode("append")
           .option("mergeSchema", "true")
           .partitionBy("partition_date")
           .save(f"{BRONZE_BASE}/delta/bronze_workflow_runs")
)

print("✅ bronze_workflow_runs written")

# COMMAND ----------

# ============================================================
# CELL 3 — Bronze: commits
# ============================================================
commits_path = f"{BRONZE_BASE}/github_commits/{REPO_PARTITION}/{DATE_PATH}/commits.json"

df_commits = (
    spark.read
         .option("multiLine", False)
         .json(commits_path)
         .withColumn("ingested_at", current_timestamp())
         .withColumn("source_repo", lit("fastapi/fastapi"))
         .withColumn("partition_date", lit(f"{YEAR}-{MONTH}-{DAY}"))
)

print(f"Rows read: {df_commits.count()}")

(
    df_commits.write
              .format("delta")
              .mode("append")
              .option("mergeSchema", "true")
              .partitionBy("partition_date")
              .save(f"{BRONZE_BASE}/delta/bronze_commits")
)

print("✅ bronze_commits written")

# COMMAND ----------

# ============================================================
# DEBUG CELL — Run this first, before writing anything
# ============================================================
pulls_path = f"{BRONZE_BASE}/github_pulls/{REPO_PARTITION}/{DATE_PATH}/pulls.json"

df_pulls = (
    spark.read
         .option("multiLine", False)
         .json(pulls_path)
)

# Print every column and its detected type
for field in df_pulls.schema.fields:
    print(f"{field.name:40s} → {field.dataType}")

# COMMAND ----------

# ============================================================
# CELL 4 — Bronze: pull_requests (full reset + rewrite)
# ============================================================
from pyspark.sql.functions import lit, current_timestamp, col, to_json

pulls_path = f"{BRONZE_BASE}/github_pulls/{REPO_PARTITION}/{DATE_PATH}/pulls.json"

# Step 1 — Delete the broken Delta table from previous attempts
dbutils.fs.rm(f"{BRONZE_BASE}/delta/bronze_pull_requests", recurse=True)
print("🗑️ Cleared old Delta table")

# Step 2 — Read fresh
df_pulls = (
    spark.read
         .option("multiLine", False)
         .json(pulls_path)
)

# Step 3 — Serialize ALL non-primitive columns to JSON string
from pyspark.sql.types import StringType, BooleanType, LongType

primitive_types = (StringType, BooleanType, LongType)

cols_to_serialize = [
    field.name for field in df_pulls.schema.fields
    if not isinstance(field.dataType, primitive_types)
]

print(f"Serializing these complex columns: {cols_to_serialize}")

for c in cols_to_serialize:
    df_pulls = df_pulls.withColumn(c, to_json(col(c)))

# Step 4 — Add metadata
df_pulls = (
    df_pulls
    .withColumn("ingested_at", current_timestamp())
    .withColumn("source_repo", lit("fastapi/fastapi"))
    .withColumn("partition_date", lit(f"{YEAR}-{MONTH}-{DAY}"))
)

print(f"Rows read: {df_pulls.count()}")

# Step 5 — Write fresh (overwrite not append, since we deleted)
(
    df_pulls.write
            .format("delta")
            .mode("overwrite")
            .partitionBy("partition_date")
            .save(f"{BRONZE_BASE}/delta/bronze_pull_requests")
)

print("✅ bronze_pull_requests written clean")

# COMMAND ----------

# ============================================================
# CELL 5 — Register Delta tables in metastore
# so you can query them with SparkSQL in Phase 5
# ============================================================
spark.sql("CREATE DATABASE IF NOT EXISTS bronze")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS bronze.workflow_runs
    USING DELTA
    LOCATION '{BRONZE_BASE}/delta/bronze_workflow_runs'
""")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS bronze.commits
    USING DELTA
    LOCATION '{BRONZE_BASE}/delta/bronze_commits'
""")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS bronze.pull_requests
    USING DELTA
    LOCATION '{BRONZE_BASE}/delta/bronze_pull_requests'
""")

print("✅ All 3 tables registered in metastore")

# COMMAND ----------

# ============================================================
# CELL 6 — Sanity checks
# ============================================================
print("=== workflow_runs ===")
spark.sql("SELECT COUNT(*) as row_count FROM bronze.workflow_runs").show()
spark.sql("SELECT id, name, status, conclusion, created_at FROM bronze.workflow_runs LIMIT 3").show(truncate=False)

print("=== commits ===")
spark.sql("SELECT COUNT(*) as row_count FROM bronze.commits").show()

print("=== pull_requests ===")
spark.sql("SELECT COUNT(*) as row_count FROM bronze.pull_requests").show()

# Check Delta history (this is the "time travel" feature)
from delta.tables import DeltaTable
dt = DeltaTable.forPath(spark, f"{BRONZE_BASE}/delta/bronze_workflow_runs")
dt.history(3).select("version", "timestamp", "operation").show()