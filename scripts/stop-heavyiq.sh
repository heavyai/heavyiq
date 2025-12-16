#!/bin/bash
# Stop HeavyIQ server container
#
# Usage:
#   ./scripts/stop-heavyiq.sh           # Stop container
#   ./scripts/stop-heavyiq.sh --clean   # Stop and remove volumes

set -e

# Navigate to project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CLEAN=""

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN="--volumes"
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean     Also remove volumes (storage data)"
            echo "  -h, --help  Show this help message"
            exit 0
            ;;
    esac
done

echo "Stopping HeavyIQ server..."

# Use newgrp to ensure docker group permissions
newgrp docker <<EOF
cd "$PROJECT_ROOT"
docker compose -f docker/docker-compose.ubuntu.yml down $CLEAN
EOF

echo "HeavyIQ server stopped."
