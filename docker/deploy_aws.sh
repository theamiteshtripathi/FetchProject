#!/bin/bash

# This script deploys the Docker container to AWS ECR and ECS
# Prerequisites:
# - AWS CLI installed and configured
# - Docker installed
# - ECR repository created
# - ECS cluster created

# Configuration
AWS_REGION="us-east-1"  # Change to your region
ECR_REPOSITORY_NAME="sentence-multitask"
ECS_CLUSTER_NAME="sentence-multitask-cluster"
ECS_SERVICE_NAME="sentence-multitask-service"
ECS_TASK_FAMILY="sentence-multitask-task"

# Build the Docker image
echo "Building Docker image..."
docker build -t sentence-multitask:latest -f docker/Dockerfile .

# Get the ECR repository URI
ECR_REPOSITORY_URI=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text)

if [ -z "$ECR_REPOSITORY_URI" ]; then
    echo "ECR repository not found. Creating..."
    ECR_REPOSITORY_URI=$(aws ecr create-repository --repository-name $ECR_REPOSITORY_NAME --region $AWS_REGION --query 'repository.repositoryUri' --output text)
fi

echo "ECR repository URI: $ECR_REPOSITORY_URI"

# Tag the Docker image
echo "Tagging Docker image..."
docker tag sentence-multitask:latest $ECR_REPOSITORY_URI:latest

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

# Push the Docker image to ECR
echo "Pushing Docker image to ECR..."
docker push $ECR_REPOSITORY_URI:latest

# Create or update the ECS task definition
echo "Creating/updating ECS task definition..."
cat > task-definition.json << EOF
{
    "family": "$ECS_TASK_FAMILY",
    "networkMode": "awsvpc",
    "executionRoleArn": "arn:aws:iam::$(aws sts get-caller-identity --query 'Account' --output text):role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "sentence-multitask",
            "image": "$ECR_REPOSITORY_URI:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 8000,
                    "hostPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/$ECS_TASK_FAMILY",
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ],
    "requiresCompatibilities": [
        "FARGATE"
    ],
    "cpu": "1024",
    "memory": "2048"
}
EOF

TASK_DEFINITION_ARN=$(aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION --query 'taskDefinition.taskDefinitionArn' --output text)
echo "Task definition ARN: $TASK_DEFINITION_ARN"

# Check if the ECS cluster exists
CLUSTER_EXISTS=$(aws ecs describe-clusters --clusters $ECS_CLUSTER_NAME --region $AWS_REGION --query 'clusters[0].clusterName' --output text)

if [ -z "$CLUSTER_EXISTS" ]; then
    echo "ECS cluster not found. Creating..."
    aws ecs create-cluster --cluster-name $ECS_CLUSTER_NAME --region $AWS_REGION
fi

# Check if the ECS service exists
SERVICE_EXISTS=$(aws ecs describe-services --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME --region $AWS_REGION --query 'services[0].serviceName' --output text)

if [ -z "$SERVICE_EXISTS" ]; then
    echo "ECS service not found. Creating..."
    aws ecs create-service \
        --cluster $ECS_CLUSTER_NAME \
        --service-name $ECS_SERVICE_NAME \
        --task-definition $TASK_DEFINITION_ARN \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$(aws ec2 describe-subnets --region $AWS_REGION --query 'Subnets[0].SubnetId' --output text)],securityGroups=[$(aws ec2 describe-security-groups --region $AWS_REGION --query 'SecurityGroups[0].GroupId' --output text)],assignPublicIp=ENABLED}" \
        --region $AWS_REGION
else
    echo "Updating ECS service..."
    aws ecs update-service \
        --cluster $ECS_CLUSTER_NAME \
        --service $ECS_SERVICE_NAME \
        --task-definition $TASK_DEFINITION_ARN \
        --region $AWS_REGION
fi

echo "Deployment completed!"

# Clean up
rm task-definition.json

echo "To check the status of the service, run:"
echo "aws ecs describe-services --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME --region $AWS_REGION" 