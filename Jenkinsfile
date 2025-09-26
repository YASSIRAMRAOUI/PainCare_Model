pipeline {
  agent any

  environment {
    COMPOSE_PROJECT_NAME = 'paincare'
    DOMAIN = credentials('PAINCARE_DOMAIN')
    CADDY_EMAIL = credentials('PAINCARE_CADDY_EMAIL')
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
          // Ensure env file exists (non-secret defaults). Jenkins Credentials can be injected to .env if needed
          sh 'test -f .env || echo DOMAIN=${DOMAIN}\nCADDY_EMAIL=${CADDY_EMAIL} > .env'

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
