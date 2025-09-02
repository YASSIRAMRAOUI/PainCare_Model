#!/bin/bash

# PainCare AI Model - Production Deployment Script
# Run this script to deploy the AI model to production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="ghcr.io"
IMAGE_NAME="your-username/paincare-ai"
VERSION=${1:-latest}
ENVIRONMENT=${2:-production}

echo -e "${GREEN}🚀 Starting PainCare AI Model Deployment${NC}"
echo -e "Registry: ${REGISTRY}"
echo -e "Image: ${IMAGE_NAME}:${VERSION}"
echo -e "Environment: ${ENVIRONMENT}"
echo ""

# Function to print status
print_status() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
fi

print_success "Prerequisites check passed"

# Build the Docker image
print_status "Building Docker image..."
docker build -t ${REGISTRY}/${IMAGE_NAME}:${VERSION} .
print_success "Docker image built successfully"

# Run tests
print_status "Running tests..."
docker run --rm ${REGISTRY}/${IMAGE_NAME}:${VERSION} python -m pytest tests/
print_success "Tests passed"

# Push to registry (optional)
if [ "$3" = "--push" ]; then
    print_status "Pushing image to registry..."
    docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    print_success "Image pushed to registry"
fi

# Deploy using docker-compose
print_status "Deploying with Docker Compose..."

# Create necessary directories
mkdir -p logs models monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources

# Copy environment file
if [ "$ENVIRONMENT" = "production" ]; then
    cp .env.production .env
else
    cp .env.development .env
fi

# Start services
docker-compose down --remove-orphans
docker-compose up -d

print_success "Services started successfully"

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."
sleep 30

# Health check
print_status "Performing health check..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Health check passed"
else
    print_error "Health check failed"
fi

# Display service status
print_status "Service Status:"
docker-compose ps

echo ""
print_success "🎉 Deployment completed successfully!"
echo ""
echo -e "${GREEN}Services available at:${NC}"
echo -e "  • API: http://localhost:8000"
echo -e "  • API Docs: http://localhost:8000/docs"
echo -e "  • Grafana: http://localhost:3000 (admin/admin123)"
echo -e "  • Prometheus: http://localhost:9090"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Configure your domain and SSL certificates"
echo -e "  2. Set up monitoring alerts"
echo -e "  3. Configure backup strategies"
echo -e "  4. Review security settings"
