# 🚀 PainCare AI Production Deployment Guide

## Quick Deployment Options

### Option 1: Docker Compose (Recommended for single server)
```bash
# 1. Clone repository
git clone <your-repo-url>
cd PainCare_Model

# 2. Configure environment
cp .env.production .env
# Edit .env with your Firebase credentials

# 3. Deploy
chmod +x deploy.sh
./deploy.sh latest production --push
```

### Option 2: Kubernetes (Recommended for scale)
```bash
# 1. Create secrets
kubectl create secret generic firebase-credentials \
  --from-file=firebase-service-account.json

# 2. Deploy
kubectl apply -f k8s-deployment.yaml

# 3. Check status
kubectl get pods -l app=paincare-ai
```

### Option 3: Cloud Platforms

#### AWS Elastic Beanstalk
```bash
eb init paincare-ai --region us-east-1
eb create production --instance-type t3.medium
eb deploy
```

#### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/paincare-ai
gcloud run deploy --image gcr.io/YOUR_PROJECT/paincare-ai --platform managed
```

#### Azure Container Instances
```bash
az container create \
  --resource-group paincare-ai \
  --name paincare-ai-api \
  --image your-registry/paincare-ai:latest \
  --dns-name-label paincare-ai \
  --ports 8000
```

## Production Checklist

### Before Deployment
- [ ] Firebase service account configured
- [ ] Environment variables set
- [ ] SSL certificates obtained
- [ ] Domain DNS configured
- [ ] Monitoring tools setup
- [ ] Backup strategy planned

### Security
- [ ] Change default passwords
- [ ] Configure CORS origins
- [ ] Enable rate limiting
- [ ] Set up API authentication
- [ ] Configure firewall rules
- [ ] Enable HTTPS only

### Monitoring
- [ ] Health checks working
- [ ] Prometheus metrics enabled
- [ ] Grafana dashboards imported
- [ ] Alerts configured
- [ ] Log aggregation setup

### Performance
- [ ] Load testing completed
- [ ] Auto-scaling configured
- [ ] CDN setup (if needed)
- [ ] Database optimized
- [ ] Caching enabled

## Scaling Guidelines

### Vertical Scaling (Single Server)
- **Small Load**: 2 CPU, 4GB RAM
- **Medium Load**: 4 CPU, 8GB RAM  
- **High Load**: 8 CPU, 16GB RAM

### Horizontal Scaling (Multiple Servers)
- **Load Balancer**: nginx/HAProxy
- **Min Replicas**: 3
- **Max Replicas**: 10
- **Auto-scale Trigger**: CPU > 70%

## Troubleshooting

### Common Issues
1. **Model not loading**: Check Firebase credentials
2. **Slow responses**: Enable Redis caching
3. **Memory issues**: Reduce batch size
4. **High CPU**: Scale horizontally

### Debug Commands
```bash
# Check logs
docker-compose logs -f paincare-ai

# Health check
curl http://localhost:8000/health/detailed

# Resource usage
docker stats

# Database connectivity
kubectl exec -it <pod-name> -- python -c "from src.services.firebase_service import FirebaseService; fs = FirebaseService(); print('Connected!' if fs.db else 'Failed!')"
```

## Support
- Documentation: [README.md](README.md)
- Issues: GitHub Issues
- Email: support@paincare.ai
