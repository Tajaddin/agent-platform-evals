.PHONY: install lint fmt typecheck test test-pg evals evals-live warehouse serve up up-obs down logs load alerts-test docker-build tf-validate teardown

UV ?= uv
COMPOSE ?= docker compose

install:
	$(UV) sync --all-extras

warehouse:
	$(UV) run python scripts/build_demo_warehouse.py

lint:
	$(UV) run ruff check src evals tests scripts
	$(UV) run ruff format --check src evals tests scripts

fmt:
	$(UV) run ruff format src evals tests scripts
	$(UV) run ruff check --fix src evals tests scripts

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest -m "not postgres and not live" --cov --cov-report=term-missing

test-pg:  ## needs DATABASE_URL (make up starts one on localhost:5432)
	$(UV) run pytest -m "postgres" --cov --cov-report=term-missing

evals:
	$(UV) run python evals/run_evals.py --provider fake --gate

evals-live:  ## costs money: needs ANTHROPIC_API_KEY
	$(UV) run python evals/run_evals.py --provider anthropic --judge --gate

serve:
	$(UV) run analyst-agent serve --port 8000

up:
	$(COMPOSE) up -d --build api postgres

up-obs:
	$(COMPOSE) --profile observability up -d --build

down:
	$(COMPOSE) --profile observability --profile load down -v

logs:
	$(COMPOSE) logs -f api

load:  ## k6 against the compose stack; writes load/results/summary.json
	$(COMPOSE) --profile load run --rm k6

alerts-test:
	docker run --rm -v "$(CURDIR)/observability:/w" -w /w --entrypoint promtool prom/prometheus:v3.5.0 check rules alert_rules.yml
	docker run --rm -v "$(CURDIR)/observability:/w" -w /w --entrypoint promtool prom/prometheus:v3.5.0 test rules alerts_test.yml

docker-build:
	docker build -t analyst-agent:local .

tf-validate:
	docker run --rm -v "$(CURDIR)/infra/aws:/w" -w /w hashicorp/terraform:1.9 fmt -check -recursive
	docker run --rm -v "$(CURDIR)/infra/aws:/w" -w /w hashicorp/terraform:1.9 init -backend=false
	docker run --rm -v "$(CURDIR)/infra/aws:/w" -w /w hashicorp/terraform:1.9 validate

teardown:  ## local stack down and every AWS resource destroyed
	$(COMPOSE) --profile observability --profile load down -v --remove-orphans
	bash infra/aws/teardown.sh
