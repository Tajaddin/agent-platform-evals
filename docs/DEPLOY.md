# Running and deploying

## Local, no Docker

```bash
uv sync --all-extras
uv run python scripts/build_demo_warehouse.py      # data/demo_warehouse.duckdb from data/sample
cp .env.example .env                                # defaults: fake provider, one local API key
uv run analyst-agent ask "How many stations are in Florida?"
uv run analyst-agent serve --port 8000              # http://localhost:8000/docs
```

To use a real model, set `AGENT_PROVIDER=anthropic` and export `ANTHROPIC_API_KEY`, or
`AGENT_PROVIDER=bedrock` with AWS credentials that may invoke the configured inference profile.

## Compose

```bash
make up          # api + postgres (memory survives restarts)
make up-obs      # + otel-collector, tempo, prometheus, alertmanager, grafana
make load        # k6, 10 virtual users for 60 s, writes load/results/summary.json
make down
```

Ports (all overridable in `.env`): API 8000, Postgres 5432, Grafana 3000 (admin / `GRAFANA_ADMIN_PASSWORD`),
Prometheus 9090, Alertmanager 9093, Tempo 3200. Set `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318`
in `.env` when the observability profile is up so the API exports spans. Raise `AGENT_RATE_LIMIT` before a
load test; the default 30/minute per key is for the public endpoint.

## AWS (manual, costs money)

Prerequisites: an AWS account with Bedrock model access granted for the Claude Haiku 4.5 inference profile
in `us-east-1`, Terraform 1.9+, the AWS CLI configured for that account.

```bash
cd infra/aws
cp terraform.tfvars.example terraform.tfvars        # github_repository, api_keys, optional allowed_cidrs
terraform init
terraform apply                                      # ~5 min; prints alb_dns_name, ecr_repository_url, github_actions_role_arn
```

Then in the GitHub repository: add the secret `AWS_ROLE_ARN` = `github_actions_role_arn`, run the `deploy`
workflow (Actions tab, "Run workflow"). It builds the image, pushes it to ECR, rolls the ECS service and
curls `/readyz` through the load balancer. The first `apply` creates the service with no image in ECR yet,
so the task fails until the first deploy run; that is expected.

Smoke test:

```bash
URL=$(terraform -chdir=infra/aws output -raw alb_dns_name)
curl -s -X POST "$URL/v1/chat" -H "X-API-Key: <one of api_keys>" -H "Content-Type: application/json" \
  -d '{"thread_id":"demo","message":"How many stations are in Florida?"}'
```

Rough cost while up: ALB about $0.75/day, Fargate 0.5 vCPU/1 GiB about $0.40/day, RDS db.t4g.micro about
$0.30/day outside the free tier, Bedrock per token. **Tear it down when done:**

```bash
make teardown        # compose down -v, then infra/aws/teardown.sh (terraform destroy)
```

## CI

`ci.yml` runs on every push and pull request: lint and format, mypy strict, pytest with a Postgres service
(coverage floor 85%), the fake-provider eval gate, promtool unit tests for the alert rules, gitleaks, an
image build that is started and asked a question, and `terraform validate`. The live eval job runs only on
manual dispatch with `live_evals` ticked and the `ANTHROPIC_API_KEY` secret present.
