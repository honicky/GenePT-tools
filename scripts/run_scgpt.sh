#!/bin/bash
# Helper script to run code in the scGPT environment (Python 3.10)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate the scGPT virtual environment
source "$PROJECT_ROOT/.venv-scgpt/bin/activate"

# Run the command passed as arguments
exec "$@"