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

## Network Configuration

### Local Development Setup
When running the AI model server locally and connecting from mobile devices:

1. **Server Configuration**: Ensure the server listens on all interfaces
   ```python
   # In src/config.py
   API_HOST = os.getenv("API_HOST", "0.0.0.0")  # Listen on all interfaces
   ```

2. **Mobile App Configuration**: Update the IP address in your mobile app
   ```bash
   # Find your PC's WiFi IP address
   ipconfig  # Windows
   ifconfig  # Linux/Mac
   
   # Update aiService.ts with your actual WiFi IP
   # Example: Replace 'localhost' with '192.168.1.100'
   ```

3. **Common Connection Issues**:
   - **IP Mismatch**: Mobile app tries wrong IP address
   - **Firewall Blocking**: Windows Firewall may block port 8000
   - **Network Interface**: Server not listening on WiFi interface

4. **Testing Connection**:
   ```bash
   # Test server accessibility from mobile device network
   curl http://YOUR_PC_IP:8000/health
   
   # Check if server is listening on correct port
   netstat -an | findstr :8000  # Windows
   netstat -an | grep :8000     # Linux/Mac
   ```

## Production Checklist

### Before Deployment
- [ ] Firebase service account configured
- [ ] Environment variables set
- [ ] Network configuration verified
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
