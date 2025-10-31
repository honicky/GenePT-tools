#!/usr/bin/env python3
"""
Extract scGPT embeddings for training datasets.

Usage:
    # Process all training files
    python scripts/extract_training_scgpt_embeddings.py

    # Process single file
    python scripts/extract_training_scgpt_embeddings.py --training-file training_001.parquet

    # Resume from checkpoint
    python scripts/extract_training_scgpt_embeddings.py --resume
"""

import argparse
import logging
from pathlib import Path
from typing import List
import json

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb not installed. Run: pip3 install duckdb")
    exit(1)


def setup_logging(log_dir: Path):
    """Configure logging to file and console."""
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'extraction.log'),
            logging.StreamHandler()
        ]
    )


def get_training_files(training_dir: Path) -> List[Path]:
    """List all training parquet files in local directory."""
    files = list(training_dir.glob("*.parquet"))
    return sorted(files)


def get_scgpt_chunks_for_dataset(scgpt_dir: Path, dataset_uuid: str) -> List[Path]:
    """Find all chunk files for a given dataset UUID."""
    pattern = f"{dataset_uuid}_chunk_*_scgpt.parquet"
    return sorted(scgpt_dir.glob(pattern))


def extract_embeddings_duckdb(
    training_file_path: Path,
    scgpt_chunks_dir: Path,
    output_path: Path
):
    """
    Extract scGPT embeddings using DuckDB for efficient joins.
    """
    logging.info(f"Processing {training_file_path}")

    # Extract dataset UUID from filename (basename without .parquet extension)
    dataset_uuid = training_file_path.stem
    logging.info(f"Dataset UUID: {dataset_uuid}")

    # Initialize DuckDB
    conn = duckdb.connect()

    # Collect all scGPT chunk files for this dataset
    chunks = get_scgpt_chunks_for_dataset(scgpt_chunks_dir, dataset_uuid)
    if not chunks:
        raise ValueError(f"No scGPT chunks found for dataset {dataset_uuid}")

    scgpt_files = [str(f) for f in chunks]
    logging.info(f"Found {len(scgpt_files)} scGPT chunk files")

    # Get row counts for validation
    training_count = conn.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{training_file_path}')
    """).fetchone()[0]

    # Perform join
    logging.info("Performing join (this may take several minutes)...")
    scgpt_files_str = "[" + ", ".join(f"'{f}'" for f in scgpt_files) + "]"

    try:
        # Execute join and write
        conn.execute(f"""
            COPY (
                SELECT
                    t.*,
                    e.* EXCLUDE (observation_joinid)
                FROM read_parquet('{training_file_path}') t
                INNER JOIN (
                    SELECT * FROM read_parquet({scgpt_files_str})
                ) e
                ON t.observation_joinid = e.observation_joinid
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    except Exception as e:
        logging.error(f"Join failed: {e}")
        # Clean up partial output
        if output_path.exists():
            output_path.unlink()
        raise

    # Validate output
    output_count = conn.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{output_path}')
    """).fetchone()[0]

    logging.info(f"Training rows: {training_count:,}")
    logging.info(f"Output rows: {output_count:,}")

    if output_count != training_count:
        missing = training_count - output_count
        pct_missing = missing / training_count * 100
        logging.warning(
            f"Row count mismatch! Training: {training_count:,}, Output: {output_count:,}"
        )
        logging.warning(f"Missing {missing:,} rows ({pct_missing:.2f}%)")

        if pct_missing > 5:
            logging.error("More than 5% of rows missing - this is unacceptable")
    else:
        logging.info("✓ Row counts match")

    conn.close()
    return output_count, training_count


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load processing checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_checkpoint(checkpoint_path: Path, checkpoint: dict):
    """Save processing checkpoint."""
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract scGPT embeddings for training datasets"
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("/mnt/scratch/cellxgene_v2_training_v1"),
        help="Directory containing training parquet files"
    )
    parser.add_argument(
        "--scgpt-chunks-dir",
        type=Path,
        default=Path("/mnt/scratch/cellxgene_v2_scgpt_chunks"),
        help="Directory containing scGPT chunk files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/scratch/cellxgene_v2_training_v1_scgpt"),
        help="Output directory for extracted embeddings"
    )
    parser.add_argument(
        "--training-file",
        help="Process single training file (filename only, not full path)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/mnt/scratch/logs"),
        help="Directory for log files"
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("="*80)
    logging.info("scGPT Training Embedding Extraction")
    logging.info("="*80)
    logging.info(f"Training Dir: {args.training_dir}")
    logging.info(f"scGPT Chunks: {args.scgpt_chunks_dir}")
    logging.info(f"Output: {args.output_dir}")
    logging.info("="*80)

    checkpoint_path = args.output_dir / "extraction_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {"completed": [], "failed": []}

    if args.resume:
        logging.info(f"Resuming from checkpoint: {len(checkpoint['completed'])} already completed")

    # Get training files to process
    if args.training_file:
        training_files = [args.training_dir / args.training_file]
    else:
        logging.info("Listing training files...")
        training_files = get_training_files(args.training_dir)
        logging.info(f"Found {len(training_files)} training files")

    # Process each training file
    total = len(training_files)
    start_time = None
    import time
    start_time = time.time()

    for idx, training_file_path in enumerate(training_files, 1):
        filename = training_file_path.name

        # Skip if already processed
        if filename in checkpoint["completed"]:
            logging.info(f"[{idx}/{total}] Skipping {filename} (already completed)")
            continue

        output_path = args.output_dir / filename

        try:
            file_start = time.time()
            logging.info("")
            logging.info(f"[{idx}/{total}] Processing {filename}")

            output_count, training_count = extract_embeddings_duckdb(
                training_file_path,
                args.scgpt_chunks_dir,
                output_path
            )

            file_time = time.time() - file_start

            # Update checkpoint
            checkpoint["completed"].append(filename)
            save_checkpoint(checkpoint_path, checkpoint)

            # Get file size
            size_mb = output_path.stat().st_size / (1024 * 1024)

            logging.info(f"✓ Completed {filename}")
            logging.info(f"  Rows: {output_count:,} / {training_count:,}")
            logging.info(f"  Size: {size_mb:.1f} MB")
            logging.info(f"  Time: {file_time/60:.1f}m")

        except Exception as e:
            logging.error(f"✗ Failed to process {filename}: {e}", exc_info=True)
            checkpoint["failed"].append({"file": filename, "error": str(e)})
            save_checkpoint(checkpoint_path, checkpoint)
            continue

    # Summary
    total_time = (time.time() - start_time) / 60 if start_time else 0

    logging.info("")
    logging.info("="*80)
    logging.info("EXTRACTION SUMMARY")
    logging.info("="*80)
    logging.info(f"Total files: {total}")
    logging.info(f"Completed: {len(checkpoint['completed'])}")
    logging.info(f"Failed: {len(checkpoint['failed'])}")
    logging.info(f"Total time: {total_time:.1f} minutes")
    logging.info("="*80)

    if checkpoint["failed"]:
        logging.info("")
        logging.info("Failed files:")
        for failure in checkpoint["failed"]:
            logging.info(f"  - {failure['file']}: {failure['error']}")

    # Exit with error code if there were failures
    if checkpoint["failed"]:
        exit(1)


if __name__ == "__main__":
    main()
