#!/bin/bash
# Start HeavyIQ server in Ubuntu container
#
# Usage:
#   ./scripts/start-heavyiq.sh          # Start in foreground (see logs)
#   ./scripts/start-heavyiq.sh -d       # Start in background (detached)
#   ./scripts/start-heavyiq.sh --build  # Force rebuild before starting

set -e

# Navigate to project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
DETACHED=""
BUILD_FLAG=""

for arg in "$@"; do
    case $arg in
        -d|--detach|--detached)
            DETACHED="-d"
            ;;
        --build)
            BUILD_FLAG="--build"
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --detach    Run in background (detached mode)"
            echo "  --build         Force rebuild the container"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
    esac
done

echo "Starting HeavyIQ server..."
echo "  - Container: heavyiq-ubuntu"
echo "  - Port: 6275"
echo "  - HeavyDB: host.docker.internal:16274"

# Use newgrp to ensure docker group permissions
newgrp docker <<EOF
cd "$PROJECT_ROOT"
docker compose -f docker/docker-compose.ubuntu.yml up $DETACHED $BUILD_FLAG
EOF

if [ -n "$DETACHED" ]; then
    echo ""
    echo "HeavyIQ server started in background."
    echo "  - API: http://localhost:6279"
    echo "  - Docs: http://localhost:6279/docs"
    echo ""
    echo "View logs: docker logs -f heavyiq-ubuntu"
    echo "Stop: ./scripts/stop-heavyiq.sh"
fi
