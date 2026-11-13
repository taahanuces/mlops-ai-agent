#!/usr/bin/env bash
# Usage: airflow_evidence.sh <dag_run_id> <out.json>   (saves dagRun + taskInstances)
set -euo pipefail
RID="$1"; OUT="$2"; BASE=${AIRFLOW_BASE_URL:-http://localhost:8080}
PY=${PY:-.venv/bin/python}
TOKEN=$(curl -s -X POST "$BASE/auth/token" -H "Content-Type: application/json" \
  -d "{\"username\":\"${AIRFLOW_USERNAME:-admin}\",\"password\":\"${AIRFLOW_PASSWORD:-admin}\"}" \
  | $PY -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
ENC=$($PY -c "import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=''))" "$RID")
RUN=$(curl -s "$BASE/api/v2/dags/training_pipeline/dagRuns/$ENC" -H "Authorization: Bearer $TOKEN")
TIS=$(curl -s "$BASE/api/v2/dags/training_pipeline/dagRuns/$ENC/taskInstances" -H "Authorization: Bearer $TOKEN")
$PY - "$OUT" <<PYEOF
import json,sys
run=json.loads('''$RUN'''); tis=json.loads('''$TIS''')
json.dump({"dag_run":run,"task_instances":[{k:t.get(k) for k in ("task_id","state","try_number","start_date","end_date","duration")} for t in tis["task_instances"]]},open(sys.argv[1],"w"),indent=2)
print("saved",sys.argv[1],run["state"])
PYEOF
