# Databricks notebook source
storage_account = "cicdriskpoornima001"

account_key = dbutils.secrets.get(
    scope="cicd-scope",
    key="adls-key"
)

configs = {
    f"fs.azure.account.key.{storage_account}.blob.core.windows.net": account_key
}

for container in ["bronze", "silver", "gold"]:
    mount_point = f"/mnt/{container}"

    try:
        dbutils.fs.mount(
            source=f"wasbs://{container}@{storage_account}.blob.core.windows.net/",
            mount_point=mount_point,
            extra_configs=configs
        )
        print(f"Mounted {container}")
    except Exception as e:
        print(f"Could not mount {container}: {e}")

# COMMAND ----------

dbutils.fs.ls("/mnt/bronze")

# COMMAND ----------

dbutils.fs.ls("/mnt/silver")

# COMMAND ----------

dbutils.fs.ls("/mnt/gold")