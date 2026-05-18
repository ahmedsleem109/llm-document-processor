#!/bin/bash
# Build, push to ECR, and update Lambda function
set -e

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_DEFAULT_REGION:-us-east-1}
REPO_NAME="llm-document-processor"
IMAGE_TAG="latest"
FUNCTION_NAME="llm-document-processor"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "Building Docker image..."
docker build -t ${REPO_NAME}:${IMAGE_TAG} ./deployment

echo "Authenticating with ECR..."
aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin \
  "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Creating ECR repo (if needed)..."
aws ecr describe-repositories --repository-names ${REPO_NAME} 2>/dev/null || \
  aws ecr create-repository --repository-name ${REPO_NAME}

echo "Tagging and pushing image..."
docker tag ${REPO_NAME}:${IMAGE_TAG} ${ECR_URI}
docker push ${ECR_URI}

echo "Updating Lambda function..."
aws lambda update-function-code \
  --function-name ${FUNCTION_NAME} \
  --image-uri ${ECR_URI}

echo "Done: Lambda updated with ${ECR_URI}"
