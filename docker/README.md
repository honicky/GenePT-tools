# Docker Utilities

This directory contains Docker configurations and utilities for the project.

## Setup for EC2 Instances

### Moving Docker Storage to /data Volume

If you're on an EC2 instance with limited root filesystem space, use the setup script to move Docker's storage to a larger volume:

```bash
sudo ./setup_docker_on_ec2.sh
```

This script:
- Stops Docker service
- Moves Docker data to `/data/docker`
- Updates Docker daemon configuration
- Restarts Docker with new storage location

After running this script, all Docker operations (builds, images, containers) will use the `/data` volume.

## Project Docker Images

### scGPT Embedding Generation

Build the scGPT Docker image for embedding generation:

```bash
cd scgpt
./build.sh
```

This will:
1. Copy required Python scripts to the build context
2. Build the Docker image with scGPT and dependencies
3. Clean up temporary files
4. Display commands for pushing to ECR

See [scgpt/README.md](scgpt/README.md) for detailed usage instructions.

## Docker Best Practices on EC2

1. **Check available space before building:**
   ```bash
   df -h /var/lib/docker  # Or /data/docker if moved
   ```

2. **Clean up unused Docker resources:**
   ```bash
   docker system prune -af
   ```

3. **Monitor Docker disk usage:**
   ```bash
   docker system df
   ```

## Directory Structure

```
docker/
├── README.md                    # This file
├── setup_docker_on_ec2.sh      # Move Docker storage to /data
└── scgpt/                       # scGPT-specific Docker files
    ├── Dockerfile
    ├── build.sh                 # Build script
    └── README.md                # scGPT documentation
```