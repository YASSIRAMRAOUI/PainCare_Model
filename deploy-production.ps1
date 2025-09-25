# PainCare AI Model - Production Deployment Script for Windows
# This script sets up the complete production environment on Windows

param(
    [switch]$SkipPrerequisites = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host @"
PainCare AI Model Production Deployment Script

USAGE:
    .\deploy-production.ps1 [OPTIONS]

OPTIONS:
    -SkipPrerequisites    Skip prerequisite checks
    -Help                 Show this help message

EXAMPLES:
    .\deploy-production.ps1                    # Full deployment with checks
    .\deploy-production.ps1 -SkipPrerequisites # Skip prerequisite validation
"@
    exit 0
}

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

Write-Host "🚀 Starting PainCare AI Model Production Deployment..." -ForegroundColor Cyan

# Check prerequisites
if (-not $SkipPrerequisites) {
    Write-Status "Checking prerequisites..."

    # Check Docker
    try {
        $dockerVersion = docker --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker not found"
        }
        Write-Success "Docker found: $dockerVersion"
    }
    catch {
        Write-Error "Docker is not installed or not in PATH. Please install Docker Desktop for Windows."
        Write-Host "Download from: https://docs.docker.com/desktop/windows/install/"
        exit 1
    }

    # Check Docker Compose
    try {
        $composeVersion = docker-compose --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose not found"
        }
        Write-Success "Docker Compose found: $composeVersion"
    }
    catch {
        Write-Error "Docker Compose is not installed or not in PATH."
        exit 1
    }

    # Check if Docker is running
    try {
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker daemon not running"
        }
        Write-Success "Docker daemon is running"
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }

    Write-Success "Prerequisites check passed"
}

# Environment setup
Write-Status "Setting up environment variables..."

if (-not (Test-Path ".env")) {
    Write-Warning ".env file not found. Creating from template..."
    
    # Generate random secret key and password
    $SecretKey = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 64 | ForEach-Object {[char]$_})
    $GrafanaPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 12 | ForEach-Object {[char]$_})
    
    @"
# PainCare AI Production Environment Variables

# Security
SECRET_KEY=$SecretKey
GRAFANA_PASSWORD=$GrafanaPassword

# Firebase Configuration
FIREBASE_SERVICE_ACCOUNT_PATH=./firebase-service-account.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com/

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# Monitoring
ENABLE_MONITORING=true
"@ | Out-File -FilePath ".env" -Encoding UTF8
    
    Write-Success ".env file created with secure defaults"
    Write-Warning "Please update the Firebase configuration in .env file"
} else {
    Write-Success ".env file found"
}

# Firebase credentials check
Write-Status "Checking Firebase credentials..."

if (-not (Test-Path "firebase-service-account.json")) {
    Write-Error "Firebase service account file not found!"
    Write-Error "Please add your firebase-service-account.json file to the project root"
    Write-Error "You can download it from your Firebase Console -> Project Settings -> Service Accounts"
    
    # Optionally open Firebase console
    $openConsole = Read-Host "Would you like to open Firebase Console now? (y/N)"
    if ($openConsole -eq 'y' -or $openConsole -eq 'Y') {
        Start-Process "https://console.firebase.google.com/"
    }
    
    exit 1
}

Write-Success "Firebase credentials found"

# Create necessary directories
Write-Status "Creating necessary directories..."
$directories = @("logs", "models", "ssl", "monitoring\grafana\dashboards", "monitoring\grafana\datasources")

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Success "Directories created"

# Install Python dependencies locally (for development)
Write-Status "Installing Python dependencies..."
try {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        python -m pip install -r requirements.txt
        Write-Success "Python dependencies installed"
    } else {
        Write-Warning "Python not found in PATH. Skipping local dependency installation."
    }
} catch {
    Write-Warning "Failed to install Python dependencies locally. Docker build will handle this."
}

# Build and start services
Write-Status "Building Docker images..."
docker-compose -f docker-compose.production.yml build
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    exit 1
}

Write-Status "Starting services..."
docker-compose -f docker-compose.production.yml up -d
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start services"
    exit 1
}

# Wait for services to be ready
Write-Status "Waiting for services to be ready..."
Start-Sleep -Seconds 30

# Health checks
Write-Status "Performing health checks..."

# Check API health
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 10 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Success "API service is healthy"
    } else {
        throw "API returned status code: $($response.StatusCode)"
    }
} catch {
    Write-Error "API service health check failed: $_"
    Write-Status "Checking API logs..."
    docker-compose -f docker-compose.production.yml logs paincare-ai-api
}

# Check Management interface
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/api/system/stats" -TimeoutSec 10 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Success "Management interface is healthy"
    } else {
        throw "Management interface returned status code: $($response.StatusCode)"
    }
} catch {
    Write-Error "Management interface health check failed: $_"
    Write-Status "Checking management logs..."
    docker-compose -f docker-compose.production.yml logs paincare-management
}

# Load environment variables for display
$envContent = Get-Content ".env" | Where-Object { $_ -match "^GRAFANA_PASSWORD=" }
$grafanaPassword = ($envContent -split "=")[1]

# Display service URLs
Write-Host ""
Write-Success "🎉 Deployment completed successfully!"
Write-Host ""
Write-Host "📊 Service URLs:" -ForegroundColor Cyan
Write-Host "   • Management Dashboard: http://localhost (or http://your-domain.com)"
Write-Host "   • AI API: http://localhost/api (or http://your-domain.com/api)"
Write-Host "   • Grafana Monitoring: http://localhost:3000"
Write-Host "   • Prometheus Metrics: http://localhost:9090"
Write-Host ""
Write-Host "🔐 Default Credentials:" -ForegroundColor Cyan
Write-Host "   • Grafana Admin: admin / $grafanaPassword"
Write-Host ""
Write-Host "📝 Important Notes:" -ForegroundColor Cyan
Write-Host "   • Update your domain in nginx.conf for production use"
Write-Host "   • Configure SSL certificates for HTTPS"
Write-Host "   • Review and adjust resource limits in docker-compose.production.yml"
Write-Host "   • Monitor logs: docker-compose -f docker-compose.production.yml logs -f"
Write-Host "   • Stop services: docker-compose -f docker-compose.production.yml down"
Write-Host ""

# Security recommendations
Write-Warning "🔒 Security Recommendations:"
Write-Host "   1. Configure Windows Firewall to restrict access to necessary ports only"
Write-Host "   2. Set up SSL certificates (Let's Encrypt recommended)"
Write-Host "   3. Configure proper backup for Firebase data and models"
Write-Host "   4. Regularly update Docker images and dependencies"
Write-Host "   5. Monitor system logs and set up alerting"
Write-Host "   6. Change default Grafana password after first login"

Write-Status "Deployment script completed!"

# Optionally open browser
$openBrowser = Read-Host "Would you like to open the management dashboard now? (y/N)"
if ($openBrowser -eq 'y' -or $openBrowser -eq 'Y') {
    Start-Process "http://localhost"
}