#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Building test container with ms-mint and ms-mint-app..."
docker compose -f docker-compose.test.yml build

echo "Running tests..."
docker compose -f docker-compose.test.yml run --rm test

echo "Done."
