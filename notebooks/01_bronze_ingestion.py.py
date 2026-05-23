# Databricks notebook source
import requests
import json
from datetime import datetime

REPOS = ["fastapi/fastapi"]

ENDPOINTS = {
    "runs":    "actions/runs",
    "commits": "commits",
    "pulls":   "pulls"
}

HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "cicd-failure-predictor-v1"
}

PER_PAGE = 100   # ← increased from 30 to GitHub max
MAX_PAGES = 5    # ← fetch 5 pages = up to 500 records per endpoint
today = datetime.utcnow()

# ── FETCH FUNCTION (now with pagination) ─────────────────────────────────────
def fetch_github(repo, endpoint_key, per_page=PER_PAGE, max_pages=MAX_PAGES):
    path = ENDPOINTS[endpoint_key]
    url  = f"https://api.github.com/repos/{repo}/{path}"
    
    all_records = []
    
    for page in range(1, max_pages + 1):
        params = {"per_page": per_page, "page": page}
        if endpoint_key == "pulls":
            params["state"] = "all"
        
        print(f"Fetching: {url} — page {page}")
        response = requests.get(url, headers=HEADERS, params=params)
        
        if response.status_code == 403:
            raise Exception("Rate limited by GitHub. Add a PAT token.")
        if response.status_code == 404:
            raise Exception(f"Repo not found: {repo}")
        if response.status_code != 200:
            raise Exception(f"GitHub API error {response.status_code}: {response.text[:200]}")
        
        data = response.json()
        
        if endpoint_key == "runs":
            records = data.get("workflow_runs", [])
        else:
            records = data  # commits and pulls return direct list
        
        if not records:
            print(f"  No more records at page {page} — stopping")
            break
        
        all_records.extend(records)
        print(f"  Page {page}: {len(records)} records (total so far: {len(all_records)})")
        
        # Stop early if GitHub returned fewer than per_page
        # means we've hit the last page
        if len(records) < per_page:
            print(f"  Last page reached")
            break
    
    return all_records

# ── WRITE FUNCTION (unchanged) ────────────────────────────────────────────────
def write_to_bronze(records, repo, endpoint_key):
    if not records:
        print(f"⚠️ No records for {repo}/{endpoint_key} — skipping")
        return
    
    repo_safe = repo.replace("/", "_")
    partition = f"year={today.year}/month={today.month:02d}/day={today.day:02d}"
    path = f"/mnt/bronze/github_{endpoint_key}/repo={repo_safe}/{partition}/{endpoint_key}.json"
    
    json_lines = "\n".join(json.dumps(record) for record in records)
    dbutils.fs.put(path, json_lines, overwrite=True)
    
    print(f"✅ {len(records)} records → {path}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
for repo in REPOS:
    for endpoint_key in ENDPOINTS:
        records = fetch_github(repo, endpoint_key)
        write_to_bronze(records, repo, endpoint_key)

print("\n🎉 Bronze ingestion complete")