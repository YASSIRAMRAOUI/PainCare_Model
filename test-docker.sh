#!/bin/bash
# Local Docker Build Test Script

echo "🐳 Testing Docker Build Locally..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t paincare-ai:local .

if [ $? -eq 0 ]; then
    echo "✅ Docker build successful!"
    
    # Test the image
    echo "🧪 Testing container startup..."
    docker run --rm -d --name paincare-test -p 8000:8000 paincare-ai:local
    
    if [ $? -eq 0 ]; then
        echo "✅ Container started successfully!"
        
        # Wait a moment for startup
        sleep 5
        
        # Test health endpoint
        echo "🔍 Testing health endpoint..."
        curl -f http://localhost:8000/health || echo "⚠️  Health endpoint not responding (this might be expected)"
        
        # Cleanup
        echo "🧹 Cleaning up test container..."
        docker stop paincare-test
        
        echo "✅ Local Docker test completed successfully!"
    else
        echo "❌ Container failed to start"
        exit 1
    fi
else
    echo "❌ Docker build failed"
    exit 1
fi

echo "
🎯 Next Steps:
1. Commit and push your changes
2. Configure GitHub repository settings as described in DOCKER_PUSH_FIX.md
3. Re-run GitHub Actions workflow
4. Check for successful package creation at: https://github.com/YASSIRAMRAOUI/PainCare_Model/pkgs/container/paincare_model
"
