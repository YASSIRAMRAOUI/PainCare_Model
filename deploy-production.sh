#!/bin/bash

# PainCare AI Model - Production Deployment Script
# This script sets up the complete production environment

set -e  # Exit on any error

echo "🚀 Starting PainCare AI Model Production Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check prerequisites
print_status "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Prerequisites check passed"

# Environment setup
print_status "Setting up environment variables..."

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from template..."
    
    # Generate random secret key
    SECRET_KEY=$(openssl rand -hex 32)
    GRAFANA_PASSWORD=$(openssl rand -base64 12)
    
    cat > .env << EOF
# PainCare AI Production Environment Variables

# Security
SECRET_KEY=${SECRET_KEY}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}

# Firebase Configuration
FIREBASE_SERVICE_ACCOUNT_PATH=./firebase-service-account.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com/

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# Monitoring
ENABLE_MONITORING=true
EOF
    
    print_success ".env file created with secure defaults"
    print_warning "Please update the Firebase configuration in .env file"
else
    print_success ".env file found"
fi

# Source environment variables
source .env

# Firebase credentials check
print_status "Checking Firebase credentials..."

if [ ! -f "firebase-service-account.json" ]; then
    print_error "Firebase service account file not found!"
    print_error "Please add your firebase-service-account.json file to the project root"
    print_error "You can download it from your Firebase Console -> Project Settings -> Service Accounts"
    exit 1
fi

print_success "Firebase credentials found"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs models ssl monitoring/grafana/dashboards monitoring/grafana/datasources

# Set proper permissions
chmod 755 logs models
print_success "Directories created"

# Build and start services
print_status "Building Docker images..."
docker-compose -f docker-compose.production.yml build

print_status "Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be healthy
print_status "Waiting for services to be ready..."
sleep 30

# Health checks
print_status "Performing health checks..."

# Check API health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_success "API service is healthy"
else
    print_error "API service health check failed"
    docker-compose -f docker-compose.production.yml logs paincare-ai-api
fi

# Check Management interface
if curl -f http://localhost:5000/api/system/stats > /dev/null 2>&1; then
    print_success "Management interface is healthy"
else
    print_error "Management interface health check failed"
    docker-compose -f docker-compose.production.yml logs paincare-management
fi

# Display service URLs
print_success "🎉 Deployment completed successfully!"
echo ""
echo "📊 Service URLs:"
echo "   • Management Dashboard: http://localhost (or http://your-domain.com)"
echo "   • AI API: http://localhost/api (or http://your-domain.com/api)"
echo "   • Grafana Monitoring: http://localhost:3000"
echo "   • Prometheus Metrics: http://localhost:9090"
echo ""
echo "🔐 Default Credentials:"
echo "   • Grafana Admin: admin / ${GRAFANA_PASSWORD}"
echo ""
echo "📝 Important Notes:"
echo "   • Update your domain in nginx.conf for production use"
echo "   • Configure SSL certificates for HTTPS"
echo "   • Review and adjust resource limits in docker-compose.production.yml"
echo "   • Monitor logs: docker-compose -f docker-compose.production.yml logs -f"
echo "   • Stop services: docker-compose -f docker-compose.production.yml down"
echo ""

# Security recommendations
print_warning "🔒 Security Recommendations:"
echo "   1. Configure firewall to restrict access to necessary ports only"
echo "   2. Set up SSL certificates (Let's Encrypt recommended)"
echo "   3. Configure proper backup for Firebase data and models"
echo "   4. Regularly update Docker images and dependencies"
echo "   5. Monitor system logs and set up alerting"
echo "   6. Change default Grafana password after first login"

print_status "Deployment script completed!"