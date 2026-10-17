#!/usr/bin/env bash
# Stops everything this repository can start: the compose stack with its volumes, and, when a Terraform
# state exists under infra/aws, every AWS resource. Safe to run twice.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose --profile observability --profile load down -v --remove-orphans || true

if [ -f infra/aws/terraform.tfstate ] || [ -d infra/aws/.terraform ]; then
  bash infra/aws/teardown.sh
else
  echo "no Terraform state under infra/aws; nothing to destroy in AWS"
fi
