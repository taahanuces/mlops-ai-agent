#!/usr/bin/env bash
# Trigger a success run and a deliberate quality-gate failure run; save evidence JSON.
set -euo pipefail
BASE=${AIRFLOW_BASE_URL:-http://localhost:8080}; PY=${PY:-.venv/bin/python}
tok() { curl -s -X POST "$BASE/auth/token" -H "Content-Type: application/json" -d "{\"username\":\"${AIRFLOW_USERNAME:-admin}\",\"password\":\"${AIRFLOW_PASSWORD:-admin}\"}" | $PY -c "import sys,json;print(json.load(sys.stdin)['access_token'])"; }
run() { # conf-json out
  T=$(tok)
  RID=$(curl -s -X POST "$BASE/api/v2/dags/training_pipeline/dagRuns" -H "Authorization: Bearer $T" -H "Content-Type: application/json" -d "{\"logical_date\": null, \"conf\": $1}" | $PY -c "import sys,json;print(json.load(sys.stdin)['dag_run_id'])")
  echo "triggered $RID"
  ENC=$($PY -c "import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=''))" "$RID")
  for i in $(seq 1 60); do
    S=$(curl -s "$BASE/api/v2/dags/training_pipeline/dagRuns/$ENC" -H "Authorization: Bearer $(tok)" | $PY -c "import sys,json;print(json.load(sys.stdin)['state'])")
    [ "$S" = success ] || [ "$S" = failed ] && break; sleep 5
  done
  bash scripts/airflow_evidence.sh "$RID" "$2"
}
T=$(tok); curl -s -X PATCH "$BASE/api/v2/dags/training_pipeline" -H "Authorization: Bearer $T" -H "Content-Type: application/json" -d '{"is_paused": false}' >/dev/null
run '{}' reports/evidence/airflow_success_run.json
run '{"min_f1": 0.999}' reports/evidence/airflow_failed_gate_run.json
