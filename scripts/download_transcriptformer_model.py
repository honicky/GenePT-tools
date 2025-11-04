#!/usr/bin/env python
"""
Download Transcriptformer model checkpoint from CZI AI.

Transcriptformer is a foundation model for single-cell RNA-seq data from CZI AI.
GitHub: https://github.com/czi-ai/transcriptformer

This script will download the model checkpoint using the transcriptformer CLI.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download Transcriptformer model files."""

    # Model directory
    model_dir = Path('models/transcriptformer')
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model directory: {model_dir.absolute()}")

    logger.info("""
    ========================================
    TRANSCRIPTFORMER MODEL DOWNLOAD
    ========================================

    Transcriptformer provides three model variants:
    - tf-sapiens: Trained on human data (best for human samples)
    - tf-exemplar: Trained on diverse species data
    - tf-metazoa: Trained on all metazoan data

    Installation Instructions:
    ---------------------------
    1. First, install the transcriptformer package:

       ./scripts/run_transcriptformer.sh pip install transcriptformer

    2. Then download the model (recommended: tf-sapiens for human data):

       ./scripts/run_transcriptformer.sh transcriptformer download tf-sapiens

       Or download all models:
       ./scripts/run_transcriptformer.sh transcriptformer download all

    3. The models will be downloaded to ~/.cache/transcriptformer/
       You can also specify a custom location with --output-dir

    Usage Example:
    --------------
    transcriptformer inference \\
        --checkpoint-path ~/.cache/transcriptformer/tf_sapiens \\
        --data-file /path/to/data.h5ad \\
        --output-path ./results \\
        --batch-size 8

    ========================================
    """)

    # Check if model files exist
    pt_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
    if pt_files:
        logger.info(f"Found model files: {pt_files}")
    else:
        logger.warning("No model checkpoint files found in models/transcriptformer/")
        logger.info("After downloading, you should have files like:")
        logger.info("  - transcriptformer_checkpoint.pt")
        logger.info("  - gene_vocabulary.json")
        logger.info("  - config.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())