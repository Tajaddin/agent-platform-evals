# 0005. Deploy on ECS Fargate behind an ALB, call Bedrock through the task role, skip the NAT gateway

Date: 2026-10-17. Status: accepted (infrastructure validated, not yet applied).

## Context

The target postings ask for AWS and for "agents in production". The deployment has to be real IaC that a
reviewer can read, cost little while a demo is up, hold no long-lived secrets in the container, and be
removable in one command.

## Decision

`infra/aws/` (Terraform, provider `aws ~> 6.0`):

- **Compute**: one ECS Fargate service (0.5 vCPU, 1 GiB) behind an application load balancer with
  `/readyz` as the health check and a deployment circuit breaker that rolls back a bad image.
- **Model access**: `AGENT_PROVIDER=bedrock`; the task role may call `bedrock:InvokeModel*` on one
  inference profile. No Anthropic key exists in the cloud at all.
- **Memory**: RDS Postgres (`db.t4g.micro`, 20 GB gp3) in private subnets, reachable only from the task
  security group; the connection string is assembled in Secrets Manager and injected as a container secret.
- **Secrets**: API keys and the database URL come from Secrets Manager via the execution role; the
  terraform variable holding the keys is `sensitive`.
- **Network**: tasks run in public subnets with public IPs and no NAT gateway. A NAT gateway is about
  $32/month before traffic, the largest avoidable line in a demo stack; the task security group only
  admits the load balancer, so the public IP is egress-only in practice.
- **Delivery**: `deploy.yml` assumes an OIDC role (no AWS keys in GitHub), pushes to ECR and rolls the
  service. `teardown.sh` destroys everything.

## Consequences

- `terraform validate` and `fmt -check` run in CI; `apply` is a manual step by the account owner after
  `terraform.tfvars` is filled in, and costs roughly $0.75/day for the ALB plus $0.016/hour for the task
  and the RDS free tier if eligible.
- HTTPS is not included: it needs a domain and an ACM certificate the repository does not own. The
  listener is HTTP on port 80 and `allowed_cidrs` can pin it to one address while demoing.
- Traces and metrics in AWS are not wired (no collector sidecar); CloudWatch has the structured JSON
  logs. The compose stack is where the observability story is complete.
