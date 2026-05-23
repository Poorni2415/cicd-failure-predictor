# Databricks notebook source
# ============================================================
# CELL 0 — Config
# ============================================================
STORAGE_ACCOUNT = "cicdriskpoornima001"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    "1u/MPlueAq8m+JqUd2qEziCI5t2D+tHUgR+bWYPKsXfd0/FMWq3hRV0IKINyPsGpFZRJz4ktb2y8+AStYFJvkA=="
)

SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD_BASE   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print("✅ Config loaded")

# COMMAND ----------

# ============================================================
# CELL 1 — Load all Silver tables
# ============================================================
from pyspark.sql.functions import *
from pyspark.sql.window import Window

df_runs = spark.read.format("delta").load(
    f"{SILVER_BASE}/delta/silver_workflow_runs"
)

df_commits = spark.read.format("delta").load(
    f"{SILVER_BASE}/delta/silver_commits"
)

df_pulls = spark.read.format("delta").load(
    f"{SILVER_BASE}/delta/silver_pull_requests"
)

print(f"Runs:    {df_runs.count()} rows")
print(f"Commits: {df_commits.count()} rows")
print(f"Pulls:   {df_pulls.count()} rows")

# COMMAND ----------

# ============================================================
# CELL 2 — Author-level historical failure rate (fixed)
# ============================================================

# Join runs with commits — removed files_changed (not in silver_commits)
df_runs_commits = (
    df_runs
    .join(
        df_commits.select("repo", "author_login", 
                          "partition_date", "additions", 
                          "deletions"),          # ← removed files_changed
        on=["repo", "partition_date"],
        how="left"
    )
)

# Author failure rate window
author_window = (
    Window
    .partitionBy("author_login")
    .orderBy(unix_timestamp(col("created_at")))
    .rowsBetween(Window.unboundedPreceding, -1)
)

df_runs_commits = (
    df_runs_commits
    .withColumn("author_failure_rate",
        round(avg(col("is_failed")).over(author_window), 4))
    .withColumn("author_total_builds",
        count(col("run_id")).over(author_window))
)

print(f"Rows after join: {df_runs_commits.count()}")
df_runs_commits.select(
    "run_id", "author_login",
    "author_failure_rate", "author_total_builds"
).show(5)

# COMMAND ----------

# ============================================================
# CELL 3 — Join PR features
# ============================================================

df_pr_features = (
    df_pulls
    .select(
        "repo",
        "partition_date",
        "is_draft",
        "is_merged",
        "pr_age_hours",
        "base_branch",
        col("author_login").alias("pr_author")
    )
)

df_full = (
    df_runs_commits
    .join(
        df_pr_features,
        on=["repo", "partition_date"],
        how="left"
    )
)

# is_main_branch flag — builds on main are higher stakes
df_full = df_full.withColumn(
    "is_main_branch",
    when(col("base_branch").isin("main", "master"), 1).otherwise(0)
)

print(f"Rows after PR join: {df_full.count()}")

# COMMAND ----------

# ============================================================
# CELL 4 — Repo-level daily stats
# ============================================================

# Daily build volume per repo
repo_daily = (
    df_runs
    .groupBy("repo", "partition_date")
    .agg(
        count("run_id").alias("daily_build_count"),
        round(avg("is_failed"), 4).alias("daily_failure_rate"),
        round(avg("duration_mins"), 2).alias("avg_duration_mins")
    )
)

df_full = df_full.join(
    repo_daily,
    on=["repo", "partition_date"],
    how="left"
)

print("Repo-level features added ✅")
df_full.select(
    "repo", "partition_date",
    "daily_build_count", "daily_failure_rate"
).show(5)

# COMMAND ----------

# ============================================================
# CELL 5 — Assemble gold_ci_failure_features
# One row = one build = one training example
# ============================================================

df_gold = (
    df_full
    .select(
        # ── Identifiers ──
        col("run_id"),
        col("repo"),
        col("partition_date"),

        # ── Target label ──
        col("is_failed").alias("label"),

        # ── Workflow features ──
        col("workflow_name"),
        col("duration_mins"),
        col("run_attempt"),
        col("day_of_week"),
        col("is_friday"),

        # ── Rolling repo failure rate (from Phase 5) ──
        col("rolling_failure_rate"),
        col("rolling_build_count"),

        # ── Commit features ──
        (coalesce(col("additions"), lit(0)) + 
 coalesce(col("deletions"), lit(0))).alias("files_changed"),

        # ── Author features ──
        col("author_login"),
        coalesce(col("author_failure_rate"), lit(0.0))
                 .alias("author_failure_rate"),
        coalesce(col("author_total_builds"), lit(0))
                 .alias("author_total_builds"),

        # ── PR features ──
        coalesce(col("is_draft"),      lit(0)).alias("is_draft"),
        coalesce(col("is_merged"),     lit(0)).alias("is_merged"),
        coalesce(col("pr_age_hours"),  lit(0.0)).alias("pr_age_hours"),
        coalesce(col("is_main_branch"),lit(0)).alias("is_main_branch"),

        # ── Repo daily features ──
        coalesce(col("daily_build_count"),  lit(1)).alias("daily_build_count"),
        coalesce(col("daily_failure_rate"), lit(0.0)).alias("daily_failure_rate"),
        coalesce(col("avg_duration_mins"),  lit(0.0)).alias("avg_duration_mins"),
    )

    # ── Only keep rows where label is known ──
    .filter(col("label").isNotNull())

    # ── Drop duplicates in case of fan-out from joins ──
    .dropDuplicates(["run_id"])
)

print(f"✅ Gold rows: {df_gold.count()}")
print(f"   Failed builds:  {df_gold.filter(col('label')==1).count()}")
print(f"   Success builds: {df_gold.filter(col('label')==0).count()}")
df_gold.printSchema()

# COMMAND ----------

# ============================================================
# CELL 6 — Write gold table
# ============================================================

(
    df_gold.write
           .format("delta")
           .mode("overwrite")
           .option("overwriteSchema", "true")
           .partitionBy("partition_date")
           .save(f"{GOLD_BASE}/delta/gold_ci_failure_features")
)

print("✅ gold_ci_failure_features written")

# COMMAND ----------

# ============================================================
# CELL 7 — Register in metastore and verify
# ============================================================
spark.sql("CREATE DATABASE IF NOT EXISTS gold")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS gold.ci_failure_features
    USING DELTA
    LOCATION '{GOLD_BASE}/delta/gold_ci_failure_features'
""")

print("✅ Gold table registered")

spark.sql("""
    SELECT 
        label,
        COUNT(*) as count,
        ROUND(AVG(rolling_failure_rate), 4) as avg_rolling_failure_rate,
        ROUND(AVG(author_failure_rate), 4)  as avg_author_failure_rate,
        ROUND(AVG(files_changed), 2)        as avg_files_changed,
        ROUND(AVG(duration_mins), 2)        as avg_duration_mins
    FROM gold.ci_failure_features
    GROUP BY label
    ORDER BY label
""").show()