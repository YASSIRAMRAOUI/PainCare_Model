pipeline {
  agent any

  environment {
    COMPOSE_PROJECT_NAME = 'paincare'
    DOMAIN = credentials('PAINCARE_DOMAIN')
    CADDY_EMAIL = credentials('PAINCARE_CADDY_EMAIL')
    // Optional additional secrets
    SECRET_KEY = credentials('PAINCARE_SECRET_KEY')
    FIREBASE_DATABASE_URL = credentials('PAINCARE_FIREBASE_DB_URL')
    # Optional: Docker registry creds if pushing to a registry
    # DOCKERHUB_USER = credentials('DOCKERHUB_USER')
    # DOCKERHUB_TOKEN = credentials('DOCKERHUB_TOKEN')
  }

  options {
    timestamps()
    ansiColor('xterm')
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Build Images') {
      steps {
        script {
          sh label: 'Docker version', script: 'docker version'
          sh 'docker compose version || docker --version'
          // Build images locally on the VM
          sh 'docker compose -f docker-compose.yml build --pull'
        }
      }
    }

    stage('Deploy') {
      steps {
        script {
          // Ensure env file exists and populate from Jenkins credentials
          sh 'test -f .env || touch .env'
          sh 'grep -q "^DOMAIN=" .env || echo DOMAIN=${DOMAIN} >> .env'
          sh 'grep -q "^CADDY_EMAIL=" .env || echo CADDY_EMAIL=${CADDY_EMAIL} >> .env'
          sh 'if [ -n "${SECRET_KEY}" ]; then grep -q "^SECRET_KEY=" .env || echo SECRET_KEY=${SECRET_KEY} >> .env; fi'
          sh 'if [ -n "${FIREBASE_DATABASE_URL}" ]; then grep -q "^FIREBASE_DATABASE_URL=" .env || echo FIREBASE_DATABASE_URL=${FIREBASE_DATABASE_URL} >> .env; fi'

          // Provide firebase service account file from Jenkins credentials
          withCredentials([file(credentialsId: 'FIREBASE_SERVICE_ACCOUNT', variable: 'FIREBASE_SA')]) {
            sh 'cp "$FIREBASE_SA" firebase-service-account.json'
          }

          // Create logs directory if missing
          sh 'mkdir -p logs'

          // Start/Update stack
          sh 'docker compose -f docker-compose.yml up -d'

          // Prune dangling images to save space (optional)
          sh 'docker image prune -f || true'
        }
      }
    }

    stage('Health Check') {
      steps {
        script {
          // Check containers health/status
          sh 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"'
          // API health endpoint via caddy
          sh 'curl -f https://$DOMAIN/api/health || curl -f http://$DOMAIN/api/health'
        }
      }
    }
  }

  post {
    success {
      echo 'Deployment successful.'
    }
    failure {
      echo 'Deployment failed. Check logs.'
      sh 'docker compose logs --no-color | tail -n 300 || true'
    }
  }
}
