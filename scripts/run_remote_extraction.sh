#!/bin/bash
# Deploy and run the scGPT extraction script on remote machine

set -e  # Exit on error

# Configuration
REMOTE_HOST="memverge-dataset-curation"
REMOTE_USER="ubuntu"
REMOTE_SCRIPT_PATH="/tmp/extract_training_scgpt_embeddings.py"
LOCAL_SCRIPT_PATH="$(dirname "$0")/extract_training_scgpt_embeddings.py"
TMUX_SESSION="scgpt-extraction"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== scGPT Training Extraction - Remote Deployment ===${NC}"
echo ""

# Check if local script exists
if [ ! -f "$LOCAL_SCRIPT_PATH" ]; then
    echo -e "${RED}ERROR: Script not found at $LOCAL_SCRIPT_PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}[1/3] Copying script to remote machine...${NC}"
scp "$LOCAL_SCRIPT_PATH" "${REMOTE_HOST}:${REMOTE_SCRIPT_PATH}"
echo -e "${GREEN}✓ Script copied${NC}"
echo ""

echo -e "${YELLOW}[2/3] Creating directories...${NC}"
ssh "$REMOTE_HOST" bash <<'EOF'
    mkdir -p /mnt/scratch/cellxgene_v2_training_v1_scgpt
    mkdir -p /mnt/scratch/logs
    echo "✓ Directories created"
EOF
echo ""

echo -e "${YELLOW}[3/3] Starting extraction in tmux session...${NC}"

# Parse command line arguments for pass-through
SCRIPT_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            SCRIPT_ARGS="$SCRIPT_ARGS --resume"
            shift
            ;;
        --training-file)
            SCRIPT_ARGS="$SCRIPT_ARGS --training-file $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--resume] [--training-file FILENAME]"
            echo ""
            echo "Options:"
            echo "  --resume              Resume from checkpoint"
            echo "  --training-file FILE  Process single file instead of all"
            echo ""
            echo "The script will be run in a tmux session named '$TMUX_SESSION'"
            echo "You can attach to it with: ssh $REMOTE_HOST -t 'tmux attach -t $TMUX_SESSION'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Start or reuse tmux session
ssh "$REMOTE_HOST" bash <<EOF
    # Kill existing session if it exists
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true

    # Start new session with the script using venv python
    tmux new-session -d -s $TMUX_SESSION "source ~/.venv/bin/activate && python $REMOTE_SCRIPT_PATH $SCRIPT_ARGS 2>&1 | tee /mnt/scratch/logs/extraction_console.log"

    echo "✓ Extraction started in tmux session '$TMUX_SESSION'"
EOF
echo ""
echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo "Commands:"
echo "  1. View live logs:  ssh $REMOTE_HOST -t 'tail -f /mnt/scratch/logs/extraction.log'"
echo "  2. Attach to tmux:  ssh $REMOTE_HOST -t 'tmux attach -t $TMUX_SESSION'"
echo "  3. Detach from tmux: Ctrl+B, then D"
echo "  4. Check status:    ssh $REMOTE_HOST 'tmux ls'"
echo ""
echo -e "${YELLOW}Showing initial log output (Ctrl+C to exit, script continues running):${NC}"
echo ""

# Tail logs for a bit to show it's working
ssh "$REMOTE_HOST" bash <<'EOF'
    # Wait for log file to be created
    for i in {1..10}; do
        if [ -f /mnt/scratch/logs/extraction.log ]; then
            break
        fi
        sleep 1
    done

    # Show initial logs
    if [ -f /mnt/scratch/logs/extraction.log ]; then
        tail -f /mnt/scratch/logs/extraction.log
    else
        echo "Log file not yet created. Check status with: tmux attach -t $TMUX_SESSION"
    fi
EOF
