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
  }

  stages {
    stage('Pre-Flight Cleanup') {
      steps {
        script {
          // Ultra-aggressive cleanup before starting
          sh 'docker system prune -a -f --volumes || true'
          sh 'docker builder prune -a -f || true'
          sh 'docker volume ls -q | xargs -r docker volume rm || true'
          sh 'docker network ls -q | grep -v "bridge\\|host\\|none" | xargs -r docker network rm || true'
          
          // Clean Jenkins workspace of old artifacts
          sh 'find . -name "*.log" -delete || true'
          sh 'find . -name "*.tmp" -delete || true'
          sh 'find . -name "__pycache__" -type d -exec rm -rf {} + || true'
          
          // Check disk space and fail early if insufficient
          sh '''
            AVAILABLE=$(df / | awk 'NR==2 {print $4}')
            if [ "$AVAILABLE" -lt 2000000 ]; then
              echo "ERROR: Insufficient disk space. Available: ${AVAILABLE}KB"
              exit 1
            fi
            echo "Available disk space: ${AVAILABLE}KB"
          '''
          sh 'df -h'
        }
      }
    }

    stage('Checkout') {
      steps {
        checkout scm
        script {
          // Remove any large files that shouldn't be there
          sh 'find . -name "*.joblib" -delete || true'
          sh 'find . -name "*.pkl" -delete || true'
          sh 'rm -rf .venv/ || true'
        }
      }
    }

    stage('Build Images') {
      steps {
        script {
          sh 'docker --version'
          sh 'docker compose version'
          
          // Create minimal .env file
          sh '''
            cat > .env << EOF
DOMAIN=${DOMAIN}
MANAGEMENT_PORT=7000
SECRET_KEY=${SECRET_KEY}
FIREBASE_DATABASE_URL=${FIREBASE_DATABASE_URL}
EOF
          '''
          
          // Build with BuildKit for better caching and reduced space usage
          sh 'export DOCKER_BUILDKIT=1'
          sh 'export COMPOSE_DOCKER_CLI_BUILD=1'
          
          // Build images sequentially with aggressive cleanup
          echo 'Building PainCare API image...'
          sh 'docker compose build --no-cache api'
          sh 'docker image prune -f'
          sh 'df -h | head -2'
          
          echo 'Building PainCare Management image...'
          sh 'docker compose build --no-cache management'
          sh 'docker image prune -f'
          sh 'df -h | head -2'
        }
      }
    }

    stage('Deploy') {
      steps {
        script {
          sh 'test -f .env || touch .env'
          sh 'grep -q "^DOMAIN=" .env || echo DOMAIN=${DOMAIN} >> .env'
          sh 'grep -q "^MANAGEMENT_PORT=" .env || echo MANAGEMENT_PORT=7000 >> .env'
          sh 'if [ -n "${SECRET_KEY}" ]; then grep -q "^SECRET_KEY=" .env || echo SECRET_KEY=${SECRET_KEY} >> .env; fi'
          sh 'if [ -n "${FIREBASE_DATABASE_URL}" ]; then grep -q "^FIREBASE_DATABASE_URL=" .env || echo FIREBASE_DATABASE_URL=${FIREBASE_DATABASE_URL} >> .env; fi'

          script {
            try {
              withCredentials([file(credentialsId: 'FIREBASE_SERVICE_ACCOUNT', variable: 'FIREBASE_SA')]) {
                sh 'cp "$FIREBASE_SA" firebase-service-account.json'
              }
            } catch (Exception e) {
              echo 'Warning: FIREBASE_SERVICE_ACCOUNT credential not found. Using default file if present.'
              sh 'test -f firebase-service-account.json || echo "{}" > firebase-service-account.json'
            }
          }

          sh 'mkdir -p logs'
          sh 'docker compose -f docker-compose.yml up -d'
          sh 'docker image prune -f || true'
        }
      }
    }

    stage('Health Check') {
      steps {
        script {
          sh 'docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Image}}"'
          sh 'curl -f https://$DOMAIN/api/health || curl -f http://$DOMAIN/api/health'
        }
      }
    }
  }

  post {
    always {
      script {
        // Final cleanup regardless of success or failure
        sh 'docker system df || true'
        sh 'docker image prune -f || true'
        sh 'docker container prune -f || true'
        sh 'df -h | head -2'
      }
    }
    success {
      echo 'Deployment successful.'
    }
    failure {
      echo 'Deployment failed. Check logs.'
      script {
        sh 'docker compose logs --tail=50 || true'
        sh 'docker ps -a || true'
      }
    }
  }
}
