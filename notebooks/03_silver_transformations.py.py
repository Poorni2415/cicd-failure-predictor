# Databricks notebook source
# ============================================================
# CELL 0 — Authenticate to ADLS
# ============================================================

STORAGE_ACCOUNT = "cicdriskpoornima001"

storage_key = "1u/MPlueAq8m+JqUd2qEziCI5t2D+tHUgR+bWYPKsXfd0/FMWq3hRV0IKINyPsGpFZRJz4ktb2y8+AStYFJvkA=="

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key
)

BRONZE_BASE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print("Storage configured ✅")

# COMMAND ----------

dbutils.fs.ls(BRONZE_BASE)

# COMMAND ----------

# ============================================================
# CELL 2 — Silver: workflow_runs
# ============================================================
from pyspark.sql.functions import (
    col, to_timestamp, when, unix_timestamp,
    round as spark_round, lit, from_unixtime
)
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType

df_runs = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/bronze_workflow_runs")

df_silver_runs = (
    df_runs
    # ── Parse timestamps ──
    .withColumn("created_at", to_timestamp(col("created_at")))
    .withColumn("updated_at", to_timestamp(col("updated_at")))
    .withColumn("run_started_at", to_timestamp(col("run_started_at")))

    # ── Compute duration in minutes ──
    .withColumn(
        "duration_mins",
        spark_round(
            (unix_timestamp(col("updated_at")) - unix_timestamp(col("run_started_at"))) / 60,
            2
        )
    )

    # ── Binary failure label (this is your ML target) ──
    .withColumn(
        "is_failed",
        when(col("conclusion") == "failure", 1)
        .when(col("conclusion") == "success", 0)
        .otherwise(None)
    )

    # ── Day of week feature ──
    .withColumn("day_of_week", 
        from_unixtime(unix_timestamp(col("created_at")), "EEEE"))
    .withColumn("is_friday",
        when(col("day_of_week") == "Friday", 1).otherwise(0))

    # ── Keep only useful columns ──
    .select(
        col("id").alias("run_id"),
        col("name").alias("workflow_name"),
        col("source_repo").alias("repo"),
        col("status"),
        col("conclusion"),
        col("is_failed"),
        col("created_at"),
        col("updated_at"),
        col("run_started_at"),
        col("duration_mins"),
        col("run_attempt"),
        col("day_of_week"),
        col("is_friday"),
        col("partition_date")
    )

    # ── Drop rows where conclusion is null (still running) ──
    .filter(col("conclusion").isNotNull())
)

print(f"Silver runs rows: {df_silver_runs.count()}")
df_silver_runs.printSchema()

# COMMAND ----------

# Check what secret scopes exist
dbutils.secrets.listScopes()

# COMMAND ----------

# ============================================================
# CELL 3 — Rolling 7-day failure rate per repo
# This is your key ML feature and top interview talking point
# ============================================================
from pyspark.sql.window import Window
from pyspark.sql.functions import avg, count

window_7d = (
    Window
    .partitionBy("repo")
    .orderBy(unix_timestamp(col("created_at")))
    .rowsBetween(-20, -1)   # 20 preceding rows = ~7 days of builds
)

df_silver_runs = (
    df_silver_runs
    .withColumn("rolling_failure_rate", 
                spark_round(avg(col("is_failed")).over(window_7d), 4))
    .withColumn("rolling_build_count",
                count(col("run_id")).over(window_7d))
)

print("Window features added ✅")
df_silver_runs.select(
    "run_id", "created_at", "is_failed", 
    "rolling_failure_rate", "rolling_build_count"
).show(10)

# COMMAND ----------

# ============================================================
# CELL 4 — Write silver_workflow_runs
# ============================================================
(
    df_silver_runs.write
                  .format("delta")
                  .mode("overwrite")
                  .option("overwriteSchema", "true")
                  .partitionBy("partition_date")
                  .save(f"{SILVER_BASE}/delta/silver_workflow_runs")
)

print("✅ silver_workflow_runs written")

# COMMAND ----------

# ============================================================
# CELL 5 — Silver: commits (struct access fix)
# ============================================================
from pyspark.sql.functions import get_json_object

df_commits = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/bronze_commits")

# Check which columns are structs vs strings
for field in df_commits.schema.fields:
    print(f"{field.name:20s} → {type(field.dataType).__name__}")

# COMMAND ----------

# ============================================================
# CELL 5 — Silver: commits (dot notation for struct columns)
# ============================================================

df_commits = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/bronze_commits")

df_silver_commits = (
    df_commits
    # ── Struct access with dot notation (not get_json_object) ──
    .withColumn("author_login",    col("author.login"))
    .withColumn("committer_login", col("committer.login"))
    .withColumn("message",         col("commit.message"))
    .withColumn("committed_at",    to_timestamp(col("commit.author.date")))

    # ── commit.verification or stats may not exist — use if present ──
    .withColumn("additions",    col("commit.additions").cast("int")
                if "additions" in [f.name for f in df_commits.select("commit.*").schema.fields]
                else lit(None).cast("int"))
    .withColumn("deletions",    col("commit.deletions").cast("int")
                if "deletions" in [f.name for f in df_commits.select("commit.*").schema.fields]
                else lit(None).cast("int"))

    .select(
        col("sha").alias("commit_sha"),
        col("source_repo").alias("repo"),
        col("author_login"),
        col("committer_login"),
        col("message"),
        col("additions"),
        col("deletions"),
        col("committed_at"),
        col("partition_date")
    )
    .filter(col("commit_sha").isNotNull())
)

print(f"Silver commits rows: {df_silver_commits.count()}")
df_silver_commits.show(3, truncate=False)

(
    df_silver_commits.write
                     .format("delta")
                     .mode("overwrite")
                     .option("overwriteSchema", "true")
                     .partitionBy("partition_date")
                     .save(f"{SILVER_BASE}/delta/silver_commits")
)

print("✅ silver_commits written")

# COMMAND ----------

# ============================================================
# CELL 6 — Silver: pull_requests
# ============================================================
df_pulls = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/bronze_pull_requests")

df_silver_pulls = (
    df_pulls
    # ── Extract user login from JSON string ──
    .withColumn("author_login", get_json_object(col("user"), "$.login"))

    # ── Parse timestamps ──
    .withColumn("created_at", to_timestamp(col("created_at")))
    .withColumn("updated_at", to_timestamp(col("updated_at")))
    .withColumn("closed_at",  to_timestamp(col("closed_at")))
    .withColumn("merged_at",  to_timestamp(col("merged_at")))

    # ── PR merge flag ──
    .withColumn("is_merged", when(col("merged_at").isNotNull(), 1).otherwise(0))

    # ── PR age in hours ──
    .withColumn("pr_age_hours",
        spark_round(
            (unix_timestamp(col("closed_at")) - unix_timestamp(col("created_at"))) / 3600,
            2
        )
    )

    # ── Extract base branch ──
    .withColumn("base_branch", get_json_object(col("base"), "$.ref"))

    .select(
        col("id").alias("pr_id"),
        col("number").alias("pr_number"),
        col("source_repo").alias("repo"),
        col("title"),
        col("state"),
        col("author_login"),
        col("base_branch"),
        col("is_merged"),
        col("draft").cast("int").alias("is_draft"),
        col("created_at"),
        col("updated_at"),
        col("closed_at"),
        col("merged_at"),
        col("pr_age_hours"),
        col("partition_date")
    )
    .filter(col("pr_id").isNotNull())
)

print(f"Silver pulls rows: {df_silver_pulls.count()}")

(
    df_silver_pulls.write
                   .format("delta")
                   .mode("overwrite")
                   .option("overwriteSchema", "true")
                   .partitionBy("partition_date")
                   .save(f"{SILVER_BASE}/delta/silver_pull_requests")
)

print("✅ silver_pull_requests written")

# COMMAND ----------

# ============================================================
# CELL 7 — Register Silver tables
# ============================================================
spark.sql("CREATE DATABASE IF NOT EXISTS silver")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS silver.workflow_runs
    USING DELTA
    LOCATION '{SILVER_BASE}/delta/silver_workflow_runs'
""")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS silver.commits
    USING DELTA
    LOCATION '{SILVER_BASE}/delta/silver_commits'
""")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS silver.pull_requests
    USING DELTA
    LOCATION '{SILVER_BASE}/delta/silver_pull_requests'
""")

print("✅ All Silver tables registered")