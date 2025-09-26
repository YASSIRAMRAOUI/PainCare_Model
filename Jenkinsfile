pipeline {
  agent any

  environment {
    COMPOSE_PROJECT_NAME = 'paincare'
    DOMAIN = 'paincare.vida-digital.tech'
    SECRET_KEY = 'change-me-in-production'
    FIREBASE_DATABASE_URL = ''
  }

  options {
    timestamps()
    timeout(time: 30, unit: 'MINUTES')
    skipDefaultCheckout()
    buildDiscarder(logRotator(numToKeepStr: '10'))
    disableConcurrentBuilds()
  }

  stages {
    stage('Pre-Flight Cleanup') {
      steps {
        echo '🧹 Starting pre-flight cleanup...'
        script {
          // Ultra-aggressive cleanup before starting
          echo 'Cleaning Docker system...'
          sh 'docker system prune -a -f --volumes || true'
          sh 'docker builder prune -a -f || true'
          sh 'docker volume ls -q | xargs -r docker volume rm || true'
          sh 'docker network ls -q | grep -v "bridge\\|host\\|none" | xargs -r docker network rm || true'
          
          echo 'Cleaning workspace artifacts...'
          sh 'find . -name "*.log" -delete || true'
          sh 'find . -name "*.tmp" -delete || true'
          sh 'find . -name "__pycache__" -type d -exec rm -rf {} + || true'
          
          echo 'Checking available disk space...'
          sh '''
            AVAILABLE=$(df / | awk 'NR==2 {print $4}')
            if [ "$AVAILABLE" -lt 2000000 ]; then
              echo "❌ ERROR: Insufficient disk space. Available: ${AVAILABLE}KB"
              exit 1
            fi
            echo "✅ Available disk space: ${AVAILABLE}KB"
          '''
          sh 'df -h'
        }
        echo '✅ Pre-flight cleanup completed'
      }
    }

    stage('Checkout & Setup') {
      steps {
        echo '📥 Checking out source code...'
        checkout scm
        script {
          echo 'Cleaning up old build artifacts...'
          sh 'find . -name "*.joblib" -delete || true'
          sh 'find . -name "*.pkl" -delete || true'
          sh 'rm -rf .venv/ || true'
          
          echo 'Displaying project structure...'
          sh 'ls -la'
          
          // Show commit info for visibility
          sh 'git log -1 --oneline || true'
        }
        echo '✅ Checkout and setup completed'
      }
    }

    stage('Build Images') {
      steps {
        echo '🔨 Starting Docker image build process...'
        script {
          echo 'Checking Docker environment...'
          sh 'docker --version'
          sh 'docker compose version'
          
          echo 'Creating environment configuration...'
          sh '''
            cat > .env << EOF
DOMAIN=${DOMAIN}
MANAGEMENT_PORT=7000
SECRET_KEY=${SECRET_KEY}
FIREBASE_DATABASE_URL=${FIREBASE_DATABASE_URL}
EOF
          '''
          
          echo 'Enabling BuildKit for optimized builds...'
          sh 'export DOCKER_BUILDKIT=1'
          sh 'export COMPOSE_DOCKER_CLI_BUILD=1'
          
          echo '🐍 Building PainCare API image...'
          sh 'docker compose build --no-cache api'
          sh 'docker image prune -f'
          sh 'df -h | head -2'
          
          echo '⚙️ Building PainCare Management image...'
          sh 'docker compose build --no-cache management'
          sh 'docker image prune -f'
          sh 'df -h | head -2'
        }
        echo '✅ All Docker images built successfully'
      }
    }

    stage('Deploy') {
      steps {
        echo '🚀 Starting deployment process...'
        script {
          echo 'Preparing environment configuration...'
          sh 'test -f .env || touch .env'
          sh 'grep -q "^DOMAIN=" .env || echo DOMAIN=${DOMAIN} >> .env'
          sh 'grep -q "^MANAGEMENT_PORT=" .env || echo MANAGEMENT_PORT=7000 >> .env'
          sh 'if [ -n "${SECRET_KEY}" ]; then grep -q "^SECRET_KEY=" .env || echo SECRET_KEY=${SECRET_KEY} >> .env; fi'
          sh 'if [ -n "${FIREBASE_DATABASE_URL}" ]; then grep -q "^FIREBASE_DATABASE_URL=" .env || echo FIREBASE_DATABASE_URL=${FIREBASE_DATABASE_URL} >> .env; fi'

          echo 'Setting up Firebase credentials...'
          script {
            try {
              withCredentials([file(credentialsId: 'FIREBASE_SERVICE_ACCOUNT', variable: 'FIREBASE_SA')]) {
                sh 'cp "$FIREBASE_SA" firebase-service-account.json'
                echo '✅ Firebase credentials loaded successfully'
              }
            } catch (Exception e) {
              echo '⚠️ Warning: FIREBASE_SERVICE_ACCOUNT credential not found. Using default file if present.'
              sh 'test -f firebase-service-account.json || echo "{}" > firebase-service-account.json'
            }
          }

          echo 'Creating required directories...'
          sh 'mkdir -p logs'
          
          echo '🐳 Starting Docker containers...'
          sh 'docker compose -f docker-compose.yml up -d'
          
          echo 'Cleaning up unused images...'
          sh 'docker image prune -f || true'
          
          echo 'Checking container status...'
          sh 'docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"'
        }
        echo '✅ Deployment completed successfully'
      }
    }

    stage('Health Check') {
      steps {
        echo '🏥 Performing health checks...'
        script {
          echo 'Checking container status...'
          sh 'docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Image}}"'
          
          echo 'Waiting for services to be ready...'
          sh 'sleep 10'
          
          echo 'Testing API health endpoint...'
          sh '''
            for i in {1..5}; do
              echo "Health check attempt $i/5..."
              if curl -f https://$DOMAIN/api/v1/health || curl -f http://$DOMAIN/api/v1/health; then
                echo "✅ Health check passed on attempt $i"
                break
              else
                echo "❌ Health check failed on attempt $i, retrying in 10 seconds..."
                sleep 10
              fi
              if [ $i -eq 5 ]; then
                echo "❌ All health check attempts failed"
                exit 1
              fi
            done
          '''
        }
        echo '✅ Health checks completed successfully'
      }
    }
  }

  post {
    always {
      echo '🧹 Running post-build cleanup...'
      script {
        echo 'Docker system status:'
        sh 'docker system df || true'
        sh 'docker image prune -f || true'
        sh 'docker container prune -f || true'
        echo 'Disk usage:'
        sh 'df -h | head -2'
      }
    }
    success {
      echo '🎉 ✅ Deployment completed successfully!'
      echo '🌐 Application is available at: https://${DOMAIN}'
      script {
        sh 'echo "🐳 Running containers:"'
        sh 'docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}" || true'
      }
    }
    failure {
      echo '❌ 💥 Deployment failed! Check the logs below for details.'
      script {
        echo '📋 Container logs (last 50 lines):'
        sh 'docker compose logs --tail=50 || true'
        echo '📋 Container status:'
        sh 'docker ps -a || true'
        echo '📋 System resources:'
        sh 'df -h || true'
        sh 'free -h || true'
      }
    }
    unstable {
      echo '⚠️ Build completed with warnings.'
    }
  }
}
