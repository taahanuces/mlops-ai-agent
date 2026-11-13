#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-"help"}
SCENARIO=${2:-""}
NAMESPACE="mlops-agent"

usage() {
    echo "Usage: $0 [break|fix] [k8s-config|data-corrupt|gate-fail]"
    exit 1
}

if [ "$ACTION" = "help" ] || [ -z "$SCENARIO" ]; then
    usage
fi

case "$SCENARIO" in
    k8s-config)
        if [ "$ACTION" = "break" ]; then
            echo "[Injected Failure 1] Patching ConfigMap to bad MODEL_PATH (/app/models/nonexistent.joblib)..."
            kubectl -n "$NAMESPACE" patch configmap mlops-inference-config \
                --type merge -p '{"data":{"MODEL_PATH":"/app/models/nonexistent.joblib"}}'
            echo "Restarting deployment to trigger CrashLoopBackOff/readiness failure..."
            kubectl -n "$NAMESPACE" rollout restart deployment/mlops-inference
            echo "Failure injected. Pods will fail readiness probe (/ready)."
        elif [ "$ACTION" = "fix" ]; then
            echo "[Fixing Failure 1] Restoring ConfigMap MODEL_PATH to /app/models/model.joblib..."
            kubectl -n "$NAMESPACE" patch configmap mlops-inference-config \
                --type merge -p '{"data":{"MODEL_PATH":"/app/models/model.joblib"}}'
            kubectl -n "$NAMESPACE" rollout restart deployment/mlops-inference
            echo "Fix applied. Awaiting healthy deployment roll..."
            kubectl -n "$NAMESPACE" rollout status deployment/mlops-inference --timeout=60s
        fi
        ;;

    data-corrupt)
        if [ "$ACTION" = "break" ]; then
            echo "[Injected Failure 2] Corrupting data/dataset.csv by dropping required target column..."
            if [ -f "data/dataset.csv" ]; then
                cp data/dataset.csv data/dataset.csv.bak
                # Drop target column or write invalid rows
                python -c "import pandas as pd; df = pd.read_csv('data/dataset.csv'); df.drop(columns=['target']).to_csv('data/dataset.csv', index=False)"
                echo "data/dataset.csv corrupted (missing 'target'). DVC status will report modified."
            else
                echo "Error: data/dataset.csv not found."
                exit 1
            fi
        elif [ "$ACTION" = "fix" ]; then
            echo "[Fixing Failure 2] Restoring data/dataset.csv via DVC checkout..."
            if [ -f "data/dataset.csv.bak" ]; then
                mv data/dataset.csv.bak data/dataset.csv
            fi
            dvc checkout data/dataset.csv.dvc || true
            echo "Fix applied. Working copy restored."
        fi
        ;;

    gate-fail)
        if [ "$ACTION" = "break" ]; then
            echo "[Injected Failure 3] Setting MIN_F1=0.999 in .env to trigger gate failure..."
            sed -i '' 's/MIN_F1=.*/MIN_F1=0.999/' .env 2>/dev/null || sed -i 's/MIN_F1=.*/MIN_F1=0.999/' .env
            echo "MIN_F1 set to 0.999. Airflow quality_gate will fail on next pipeline execution."
        elif [ "$ACTION" = "fix" ]; then
            echo "[Fixing Failure 3] Restoring MIN_F1=0.85 in .env..."
            sed -i '' 's/MIN_F1=.*/MIN_F1=0.85/' .env 2>/dev/null || sed -i 's/MIN_F1=.*/MIN_F1=0.85/' .env
            echo "MIN_F1 restored to 0.85."
        fi
        ;;

    *)
        echo "Unknown scenario: $SCENARIO"
        usage
        ;;
esac
