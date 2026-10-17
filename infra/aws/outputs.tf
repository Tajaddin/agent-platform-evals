output "alb_dns_name" {
  description = "Public URL of the API (HTTP)"
  value       = "http://${aws_lb.this.dns_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.this.repository_url
}

output "github_actions_role_arn" {
  description = "Store as the AWS_ROLE_ARN repository secret for the deploy workflow"
  value       = aws_iam_role.github_actions.arn
}

output "ecs_cluster" {
  value = aws_ecs_cluster.this.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.this.name
}
