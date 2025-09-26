#!/bin/bash

# Docker cleanup script to free disk space before builds
echo "=== Docker System Cleanup ==="
echo "Current disk usage:"
df -h

echo -e "\n=== Docker Images Before Cleanup ==="
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo -e "\n=== Cleaning up Docker system ==="

# Remove all stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove all unused networks
echo "Removing unused networks..."
docker network prune -f

# Remove all unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove all unused images
echo "Removing unused images..."
docker image prune -a -f

# Remove build cache
echo "Removing build cache..."
docker builder prune -f --all

# Clean up system (everything unused)
echo "Final system cleanup..."
docker system prune -a -f --volumes

echo -e "\n=== Docker Images After Cleanup ==="
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo -e "\n=== Final disk usage ==="
df -h

echo -e "\nCleanup completed!"