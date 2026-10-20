.PHONY: help install test lint eda train evaluate explain benchmark run docker-build docker-up clean

help:
	@echo "ShieldML — Developer Automation Commands"
	@echo "  make install      - Install all dependencies in editable mode"
	@echo "  make test         - Run full pytest test suite"
	@echo "  make lint         - Run ruff linter"
	@echo "  make eda          - Run exploratory data analysis and generate charts"
	@echo "  make train        - Execute full end-to-end model training pipeline"
	@echo "  make evaluate     - Run standalone model evaluation on test data"
	@echo "  make explain      - Generate local SHAP waterfall explanation figures"
	@echo "  make benchmark    - Benchmark API latency (<50ms target)"
	@echo "  make run          - Launch FastAPI prediction service locally"
	@echo "  make docker-build - Build production Docker image"
	@echo "  make docker-up    - Launch Docker Compose stack"
	@echo "  make clean        - Clean temporary files and caches"

install:
	pip install -e . --no-deps

test:
	python3 -m pytest tests/

lint:
	python3 -m ruff check src/ app/ scripts/ tests/

eda:
	python3 scripts/run_eda.py

train:
	python3 scripts/train.py

evaluate:
	python3 scripts/evaluate.py

explain:
	python3 scripts/explain.py

benchmark:
	python3 scripts/benchmark_api.py http://127.0.0.1:8000

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t shieldml:latest .

docker-up:
	docker-compose up -d

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache src/*.egg-info
