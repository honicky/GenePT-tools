#!/bin/bash
# Helper script to run code in the Transcriptformer environment (Python 3.11)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate the Transcriptformer virtual environment
source "$PROJECT_ROOT/.venv-transcriptformer/bin/activate"

# Run the command passed as arguments
exec "$@"