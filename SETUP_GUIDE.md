# GenePT Data Curation Machine Setup Guide

This guide provides step-by-step instructions for setting up a fresh machine for GenePT data curation and analysis.
Most of these are standard setups, this file just collects them all here in case you forgot how to do them.

## System Requirements

- **OS**: Ubuntu/Linux (tested on Ubuntu with Linux 6.14.0)
- **Python**: 3.10 (required for scGPT compatibility)
- **Storage**: At least 200GB free space for datasets
- **RAM**: 16GB minimum, 32GB+ recommended for large datasets

## 1. Initial System Setup

### 1.1 Create Data Directory Structure
```bash
# Create main data directory
sudo mkdir -p /data
sudo chown $USER:$USER /data
cd /data
```

### 1.2 Clone Repository
```bash
git clone https://github.com/honicky/GenePT-tools.git
cd GenePT-tools/
```

## 2. Development Tools Installation

### 2.1 Install UV Package Manager
```bash
# Install UV for Python package management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH
source $HOME/.local/bin/env

# Verify installation
uv --version
```

### 2.2 Install Node.js and NPM (for Claude Code)
```bash
# Install NVM (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# Load NVM
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Install latest Node.js
nvm install node

# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code
```

## 3. Python Environment Setup

### 3.1 Install Python Dependencies
```bash
cd /data/GenePT-tools

# Install base dependencies
uv sync

# Install development tools (includes JupyterLab, pytest, formatters)
uv sync --extra dev

# Install package in editable mode
uv pip install -e '.[dev]'
```

### 3.2 Activate Virtual Environment (if needed)
```bash
# UV automatically manages the environment, but if you need manual activation:
source /data/GenePT-tools/.venv/bin/activate
```

## 4. Cloud Services Configuration

### 4.1 AWS Configuration
```bash
# Create AWS config directory
mkdir -p ~/.aws

# Create AWS config file
cat > ~/.aws/config << 'EOF'
[default]
region = us-east-1
output = json
EOF

# Create AWS credentials file
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
EOF

# Set proper permissions
chmod 600 ~/.aws/credentials
```

### 4.2 API Keys Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file and add your keys:
# OPENAI_API_KEY=your_openai_key_here
# HF_TOKEN=your_huggingface_token_here
# HF_WRITE_TOKEN=your_huggingface_write_token_here
# WANDB_API_KEY=your_wandb_key_here
```

## 5. Data Directories Setup

### 5.1 Create Required Directories
```bash
# Create data storage directories
mkdir -p data/cellxgene
mkdir -p data/GenePT_embedding_v2
mkdir -p save
mkdir -p img
```

### 5.2 Download Initial Datasets (Optional)
```bash
# Example: Download Tabula Sapiens or other datasets
# These will be downloaded automatically when running notebooks
```

## 6. Development Environment

### 6.1 Start JupyterLab Server
```bash
# Start JupyterLab (with uv)
uv run python -m jupyterlab --no-browser --port=8888

# Or with specific options
uv run jupyter lab --port=8888 --no-browser --ip=0.0.0.0
```

### 6.2 Code Formatting Tools
```bash
# Format Python code
uv run yapf --in-place --recursive src/ test/

# Sort imports
uv run isort --gitignore .

# Format notebooks
uv run black --target-version py310 notebooks/
```

### 6.3 Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest test/test_inference.py -v
```

## 7. Optional Tools

### 7.1 DuckDB (for data analysis)
```bash
# Already included in dev dependencies
# Run DuckDB CLI
uv run duckdb
```

### 7.2 Additional Python Packages
```bash
# Install any additional packages as needed
uv pip install package_name
```

## 8. Verification Checklist

Run these commands to verify your setup:

```bash
# Check Python version
python --version  # Should be 3.10.x

# Check UV installation
uv --version

# Check virtual environment
which python  # Should point to .venv/bin/python

# Check key dependencies
uv pip list | grep -E "torch|anndata|openai|pandas|scanpy"

# Test imports
python -c "import torch, anndata, openai, pandas, scanpy; print('All imports successful')"

# Check AWS access (if configured)
aws s3 ls  # Should list buckets if configured

# Check git status
git status
```

## 9. Common Workflows

### 9.1 Running Analysis Notebooks
1. Start JupyterLab: `uv run jupyter lab`
2. Open browser at provided URL
3. Start with `notebooks/notebook_setup.ipynb`
4. Follow notebook documentation in `notebooks/README.md`

### 9.2 Processing Large Datasets
```bash
# Use the optimized shuffle script for large embeddings
uv run python scripts/shuffle_embeddings_optimized.py --help
```

### 9.3 Generating Embeddings
1. Configure API keys in `.env`
2. Run `notebooks/generate_genept_embeddings.ipynb`
3. Upload to HuggingFace using `notebooks/create_hf_repos.ipynb`

## 10. Troubleshooting

### Common Issues and Solutions

1. **UV command not found**
   ```bash
   source $HOME/.local/bin/env
   # Or add to .bashrc/.zshrc
   ```

2. **Python version mismatch**
   ```bash
   # UV will handle Python 3.10 automatically
   uv python pin 3.10
   ```

3. **Memory issues with large datasets**
   - Use `AnnDataChunker` for processing
   - Increase swap space if needed
   - Use sparse matrix operations

4. **JupyterLab kernel issues**
   ```bash
   uv run python -m ipykernel install --user --name genept
   ```

## 11. Security Notes

- Never commit `.env` file or AWS credentials
- Keep API keys secure and rotate regularly
- Use read-only tokens when possible
- Review code before running from notebooks

## 12. Next Steps

1. Review `CLAUDE.md` for project-specific guidelines
2. Explore notebooks in order (see `notebooks/README.md`)
3. Run tests to ensure everything works
4. Start with small datasets before scaling up

## Additional Resources

- [GenePT Paper](link_to_paper)
- [Project README](README.md)
- [Notebook Documentation](notebooks/README.md)
- [UV Documentation](https://docs.astral.sh/uv/)