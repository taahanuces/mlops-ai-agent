.PHONY: setup mlflow airflow-up airflow-down seed build-deploy agent test lint format demo break-k8s fix-k8s break-data fix-data clean

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
RUFF ?= $(VENV)/bin/ruff
PYTEST ?= $(VENV)/bin/pytest
UVICORN ?= $(VENV)/bin/uvicorn

setup:
	@bash scripts/setup_local.sh

mlflow:
	@mkdir -p mlruns
	@$(VENV)/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

airflow-up:
	@docker compose up -d

airflow-down:
	@docker compose down

seed:
	@$(PYTHON) scripts/seed_experiments.py

build-deploy:
	@docker build -t mlops-inference:1.0.0 .
	@kind load docker-image mlops-inference:1.0.0 --name mlops
	@kubectl apply -f kubernetes/

agent:
	@$(UVICORN) agent.main:app --host 0.0.0.0 --port 9000 --reload

test:
	@$(PYTEST) -m "not integration" --cov=agent --cov-report=term-missing

lint:
	@$(RUFF) check .
	@$(RUFF) format --check .

format:
	@$(RUFF) format .
	@$(RUFF) check --fix .

demo:
	@$(PYTHON) scripts/demo_scenarios.py

break-k8s:
	@bash scripts/inject_failure.sh break k8s-config

fix-k8s:
	@bash scripts/inject_failure.sh fix k8s-config

break-data:
	@bash scripts/inject_failure.sh break data-corrupt

fix-data:
	@bash scripts/inject_failure.sh fix data-corrupt

clean:
	@rm -rf .pytest_cache htmlcov .coverage .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
