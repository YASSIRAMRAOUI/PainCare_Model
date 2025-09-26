pipeline {
  agent any

  environment {
    COMPOSE_PROJECT_NAME = 'paincare'
    DOMAIN = credentials('PAINCARE_DOMAIN')
    CADDY_EMAIL = credentials('PAINCARE_CADDY_EMAIL')
    SECRET_KEY = credentials('PAINCARE_SECRET_KEY')
    FIREBASE_DATABASE_URL = credentials('PAINCARE_FIREBASE_DB_URL')
  }

  options {
    timestamps()
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
          sh 'test -f .env || touch .env'
          sh 'grep -q "^DOMAIN=" .env || echo DOMAIN=${DOMAIN} >> .env'
          sh 'grep -q "^CADDY_EMAIL=" .env || echo CADDY_EMAIL=${CADDY_EMAIL} >> .env'
          sh 'grep -q "^MANAGEMENT_PORT=" .env || echo MANAGEMENT_PORT=7000 >> .env'
          sh 'if [ -n "${SECRET_KEY}" ]; then grep -q "^SECRET_KEY=" .env || echo SECRET_KEY=${SECRET_KEY} >> .env; fi'
          sh 'if [ -n "${FIREBASE_DATABASE_URL}" ]; then grep -q "^FIREBASE_DATABASE_URL=" .env || echo FIREBASE_DATABASE_URL=${FIREBASE_DATABASE_URL} >> .env; fi'

          withCredentials([file(credentialsId: 'FIREBASE_SERVICE_ACCOUNT', variable: 'FIREBASE_SA')]) {
            sh 'cp "$FIREBASE_SA" firebase-service-account.json'
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
    success {
      echo 'Deployment successful.'
    }
    failure {
      echo 'Deployment failed. Check logs.'
      sh 'docker compose logs --no-color | tail -n 300 || true'
    }
  }
}
