#!/bin/bash
# Script to diagnose and fix Docker service issues on EC2

set -e

echo "=== Diagnosing Docker Service Issues ==="
echo

# 1. Check Docker daemon configuration
echo "1. Checking Docker daemon configuration..."
if [ -f /etc/docker/daemon.json ]; then
    echo "   Found daemon.json:"
    cat /etc/docker/daemon.json
    echo
    
    # Validate JSON
    if ! python3 -m json.tool /etc/docker/daemon.json > /dev/null 2>&1; then
        echo "   ❌ Invalid JSON in daemon.json!"
        echo "   Backing up and removing..."
        sudo mv /etc/docker/daemon.json /etc/docker/daemon.json.backup
        echo "   Backed up to /etc/docker/daemon.json.backup"
    fi
else
    echo "   No daemon.json found (using defaults)"
fi

# 2. Check disk space
echo
echo "2. Checking disk space..."
df -h / /var /data 2>/dev/null || df -h /

# 3. Check if Docker root directory exists and is accessible
echo
echo "3. Checking Docker root directory..."
DOCKER_ROOT="/var/lib/docker"
if [ -f /etc/docker/daemon.json ]; then
    DOCKER_ROOT=$(python3 -c "import json; print(json.load(open('/etc/docker/daemon.json')).get('data-root', '/var/lib/docker'))" 2>/dev/null || echo "/var/lib/docker")
fi
echo "   Docker root: $DOCKER_ROOT"

if [ ! -d "$DOCKER_ROOT" ]; then
    echo "   ❌ Docker root directory doesn't exist!"
    echo "   Creating directory..."
    sudo mkdir -p "$DOCKER_ROOT"
    sudo chmod 755 "$DOCKER_ROOT"
elif [ ! -w "$DOCKER_ROOT" ]; then
    echo "   ❌ Docker root directory is not writable!"
    echo "   Fixing permissions..."
    sudo chmod 755 "$DOCKER_ROOT"
else
    echo "   ✓ Directory exists and is accessible"
fi

# 4. Check for conflicting Docker processes
echo
echo "4. Checking for conflicting processes..."
if pgrep dockerd > /dev/null; then
    echo "   Found running dockerd processes:"
    ps aux | grep dockerd | grep -v grep
    echo "   Killing stale processes..."
    sudo pkill -9 dockerd || true
    sleep 2
fi

# 5. Clean up Docker socket
echo
echo "5. Cleaning up Docker socket..."
sudo rm -f /var/run/docker.sock
sudo rm -f /var/run/docker.pid

# 6. Reset Docker to minimal configuration
echo
echo "6. Resetting Docker configuration..."
cat << 'EOF' | sudo tee /etc/docker/daemon.json
{
    "storage-driver": "overlay2",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    }
}
EOF

# 7. Start Docker service
echo
echo "7. Starting Docker service..."
sudo systemctl daemon-reload
sudo systemctl start docker

# Wait for Docker to be ready
echo "   Waiting for Docker to be ready..."
for i in {1..30}; do
    if docker version > /dev/null 2>&1; then
        echo "   ✓ Docker is running!"
        break
    fi
    echo -n "."
    sleep 1
done
echo

# 8. Verify Docker is working
echo
echo "8. Verifying Docker..."
docker version
echo
docker info | grep -E "Server Version|Storage Driver|Docker Root Dir"

echo
echo "=== Docker Service Fixed ==="
echo
echo "Next steps:"
echo "1. If you need Docker on /data due to space constraints, run:"
echo "   sudo ./docker/setup_docker_on_ec2.sh"
echo
echo "2. To build with cache on /data:"
echo "   ./docker/build_with_cache.sh docker/scgpt scgpt-embeddings"