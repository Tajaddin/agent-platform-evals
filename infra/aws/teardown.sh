#!/usr/bin/env bash
# Destroys everything infra/aws created (ECS service, ALB, RDS, ECR images, secrets, IAM roles, VPC).
# Run it when the demo is over: the ALB and RDS instance bill by the hour.
set -euo pipefail
cd "$(dirname "$0")"
terraform init -input=false
terraform destroy -auto-approve -input=false
echo "all AWS resources for analyst-agent destroyed"
