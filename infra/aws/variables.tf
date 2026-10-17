variable "name" {
  description = "Resource name prefix"
  type        = string
  default     = "analyst-agent"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "image" {
  description = "Container image to run. After the first `deploy` workflow run this is <ecr url>:latest."
  type        = string
  default     = ""
}

variable "cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)"
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "bedrock_model_id" {
  description = "Bedrock inference profile or model id the task may invoke"
  type        = string
  default     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "api_keys" {
  description = "Accepted X-API-Key values for the public endpoint; stored in Secrets Manager, never in state outputs"
  type        = list(string)
  sensitive   = true
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to reach the load balancer"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "github_repository" {
  description = "owner/name of the GitHub repository allowed to assume the deploy role via OIDC"
  type        = string
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "log_retention_days" {
  type    = number
  default = 14
}
