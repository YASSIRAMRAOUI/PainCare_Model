# 🔒 Security Setup Guide

## ⚠️ IMPORTANT: Before Making Repository Public

**NEVER commit these files to GitHub:**
- `.env` files (contain API keys, secrets)
- `firebase-service-account.json` (contains private keys)
- Any file with real credentials

## 🛠️ Setup Instructions

### 1. Configure Environment Variables
```bash
# Copy example file
cp .env.example .env

# Edit .env with your actual values
nano .env
```

### 2. Setup Firebase Credentials
```bash
# Copy example file
cp firebase-service-account.example.json firebase-service-account.json

# Replace with your actual Firebase service account JSON
# Download from: Firebase Console > Project Settings > Service Accounts
```

### 3. Verify Security
```bash
# Check what will be committed (should NOT include .env or firebase-service-account.json)
git status
git add .
git status

# If you see .env or firebase-service-account.json in staging, run:
git reset .env firebase-service-account.json
```

## 🚀 Production Deployment

### Option 1: Environment Variables (Recommended)
```bash
# Set environment variables directly
export FIREBASE_SERVICE_ACCOUNT_PATH="/path/to/service-account.json"
export SECRET_KEY="your-production-secret"
export API_HOST="0.0.0.0"
```

### Option 2: Docker Secrets
```bash
# Create secrets
echo "your-secret-key" | docker secret create api_secret_key -
echo "$(cat firebase-service-account.json)" | docker secret create firebase_creds -
```

### Option 3: Kubernetes Secrets
```bash
# Create secret from file
kubectl create secret generic firebase-credentials \
  --from-file=firebase-service-account.json

# Create secret from literal
kubectl create secret generic api-secrets \
  --from-literal=SECRET_KEY="your-secret-key"
```

## 🔧 Development vs Production

### Development (.env)
```env
DEBUG_MODE=True
LOG_LEVEL=DEBUG
SECRET_KEY=dev-key-not-secure
```

### Production (Environment Variables)
```bash
export DEBUG_MODE=False
export LOG_LEVEL=INFO  
export SECRET_KEY="super-secure-production-key"
```

## ✅ Security Checklist

Before going public, ensure:
- [ ] `.env` is in `.gitignore`
- [ ] `firebase-service-account.json` is in `.gitignore`
- [ ] Example files are provided (`.env.example`)
- [ ] README mentions security setup
- [ ] No real credentials in code
- [ ] Production uses environment variables
- [ ] Secrets are rotated regularly

## 🚨 What to Do If Credentials Are Exposed

### If you accidentally commit credentials:

1. **Immediately revoke credentials:**
   - Firebase: Generate new service account
   - API keys: Regenerate all keys

2. **Remove from git history:**
```bash
# Remove file from git history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch firebase-service-account.json' \
  --prune-empty --tag-name-filter cat -- --all

# Force push (DANGEROUS - only if repo is private or just created)
git push origin --force --all
```

3. **Update all systems with new credentials**

## 📞 Need Help?
- Check `.env.example` for required variables
- See `firebase-service-account.example.json` for structure
- Read deployment guides in `DEPLOYMENT.md`
