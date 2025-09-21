#!/bin/bash
# Script to configure Docker to use /data volume on EC2

set -e

echo "=== Configuring Docker to use /data volume ==="
echo

# Check current Docker storage
echo "Current Docker storage usage:"
docker system df
echo

# Stop Docker service
echo "Stopping Docker service..."
sudo systemctl stop docker
sudo systemctl stop docker.socket

# Create new Docker directory on /data
echo "Creating Docker directory on /data volume..."
sudo mkdir -p /data/docker

# Move existing Docker data (optional - can skip if starting fresh)
echo "Moving existing Docker data (this may take a while)..."
if [ -d "/var/lib/docker" ]; then
    sudo rsync -avxP /var/lib/docker/ /data/docker/ || true
fi

# Configure Docker to use new location
echo "Configuring Docker daemon..."
sudo mkdir -p /etc/docker

# Create or update daemon.json
cat << 'EOF' | sudo tee /etc/docker/daemon.json
{
    "data-root": "/data/docker",
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# Clean up old Docker directory (optional)
echo "Cleaning up old Docker directory..."
sudo rm -rf /var/lib/docker || true

# Create symlink for compatibility (optional)
sudo ln -s /data/docker /var/lib/docker || true

# Start Docker service
echo "Starting Docker service..."
sudo systemctl daemon-reload
sudo systemctl start docker

# Verify new location
echo
echo "Verifying new Docker location..."
docker info | grep "Docker Root Dir"

# Clean up unused data
echo
echo "Cleaning up unused Docker data..."
docker system prune -af --volumes || true

echo
echo "Docker storage usage after cleanup:"
docker system df

echo
echo "Disk usage:"
df -h /data
df -h /

echo
echo "=== Configuration complete ==="
echo "Docker is now using /data/docker for storage"