################################################################################
# AutonomDS — Makefile
# Convenient developer targets for local development, testing, and deployment.
# Usage: make <target>
################################################################################

.PHONY: help dev api worker frontend test lint format typecheck clean \
        docker-up docker-down docker-build migrate docs

# ── Colors ────────────────────────────────────────────────────────────────────
CYAN  := \033[0;36m
GREEN := \033[0;32m
RESET := \033[0m

help: ## Show this help message
	@echo ""
	@echo "  $(CYAN)AutonomDS$(RESET) — Autonomous Multi-Agent Data Science Platform"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Local Development ─────────────────────────────────────────────────────────

dev: ## Start all services (API + frontend) without Docker
	@echo "$(CYAN)Starting AutonomDS locally...$(RESET)"
	@$(MAKE) -j2 api frontend

api: ## Start the FastAPI backend
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Start the Celery background worker
	celery -A app.api.main.celery_app worker \
	    --loglevel=info --queues=pipeline --concurrency=2

frontend: ## Start the Streamlit dashboard
	streamlit run app/frontend/streamlit_app.py --server.port 8501

mlflow-ui: ## Start the MLflow tracking UI
	mlflow ui --host 0.0.0.0 --port 5000 \
	    --backend-store-uri file://$(PWD)/mlflow/mlruns

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run the full test suite with coverage
	pytest tests/ -v --tb=short \
	    --cov=app --cov-report=term-missing --cov-report=html \
	    --cov-fail-under=60

test-unit: ## Run unit tests only (fast, no external services)
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	pytest tests/integration/ -v --tb=short

test-ci: ## CI-optimised test run (no coverage overhead)
	pytest tests/ --tb=short -q

# ── Code Quality ──────────────────────────────────────────────────────────────

lint: ## Lint with ruff
	ruff check app/ tests/

lint-fix: ## Auto-fix ruff linting issues
	ruff check app/ tests/ --fix

format: ## Format code with black
	black app/ tests/

format-check: ## Check formatting without modifying files
	black app/ tests/ --check

typecheck: ## Run mypy type checking
	mypy app/ --ignore-missing-imports --no-strict-optional

quality: lint format-check typecheck ## Run all quality checks

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build: ## Build all Docker images
	docker-compose build

docker-up: ## Start all services with Docker Compose
	docker-compose up -d
	@echo "$(GREEN)Services started:$(RESET)"
	@echo "  API:      http://localhost:8000"
	@echo "  Docs:     http://localhost:8000/docs"
	@echo "  Frontend: http://localhost:8501"
	@echo "  Flower:   http://localhost:5555"

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## Tail Docker Compose logs
	docker-compose logs -f

docker-restart: ## Rebuild and restart containers
	docker-compose down && docker-compose build && docker-compose up -d

# ── Environment ───────────────────────────────────────────────────────────────

env: ## Copy .env.example to .env (first-time setup)
	@if [ ! -f .env ]; then \
	    cp .env.example .env; \
	    echo "$(GREEN)Created .env from .env.example — edit values as needed$(RESET)"; \
	else \
	    echo ".env already exists — skipping"; \
	fi

install: ## Install Python dependencies
	pip install -r requirements.txt

install-dev: ## Install dependencies including dev tools
	pip install -r requirements.txt
	pip install black mypy ruff pytest-cov

# ── Directories ───────────────────────────────────────────────────────────────

dirs: ## Create required runtime directories
	mkdir -p datasets experiments uploads reports models mlflow/mlruns

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"      -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc"               -delete 2>/dev/null || true
	find . -name ".coverage"           -delete 2>/dev/null || true
	@echo "$(GREEN)Cleaned.$(RESET)"

clean-data: ## Remove all uploaded datasets and experiment outputs (CAREFUL)
	@echo "$(CYAN)Removing experiment data...$(RESET)"
	rm -rf uploads/* experiments/* reports/* models/*
	@echo "$(GREEN)Done.$(RESET)"

# ── Documentation ────────────────────────────────────────────────────────────

docs: ## Build documentation (placeholder — extend with mkdocs if needed)
	@echo "$(CYAN)Docs are in the docs/ directory.$(RESET)"
	@ls docs/
