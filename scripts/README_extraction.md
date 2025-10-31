# scGPT Training Embedding Extraction

Automated deployment and execution of scGPT embedding extraction on remote machine.

## Prerequisites (One-Time Setup)

On the remote machine (`memverge-dataset-curation`), ensure you have:

```bash
# Create virtual environment
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# Install dependencies
pip install duckdb

# Verify data directories exist
ls /mnt/scratch/cellxgene_v2_training_v1/  # Training files
ls /mnt/scratch/cellxgene_v2_scgpt_chunks/ # scGPT embeddings
```

## Quick Start

```bash
# Deploy and run the extraction (processes all training files)
./scripts/run_remote_extraction.sh

# Or with resume capability (picks up where it left off)
./scripts/run_remote_extraction.sh --resume

# Or process a single file for testing
./scripts/run_remote_extraction.sh --training-file your_file.parquet
```

## What It Does

1. **Copies** the extraction script to `memverge-dataset-curation`
2. **Creates** output directories if needed
3. **Starts** extraction in a tmux session called `scgpt-extraction` using `~/.venv`
4. **Shows** initial log output (you can Ctrl+C to disconnect, script continues)

## Monitoring

### View Live Logs
```bash
ssh memverge-dataset-curation -t 'tail -f /mnt/scratch/logs/extraction.log'
```

### Attach to tmux Session
```bash
ssh memverge-dataset-curation -t 'tmux attach -t scgpt-extraction'
# Press Ctrl+B, then D to detach without stopping the script
```

### Check tmux Sessions
```bash
ssh memverge-dataset-curation 'tmux ls'
```

### Download Checkpoint (to see progress)
```bash
scp memverge-dataset-curation:/mnt/scratch/cellxgene_v2_training_v1_scgpt/extraction_checkpoint.json .
cat extraction_checkpoint.json | python3 -m json.tool
```

## Output Location

Files are written to: `/mnt/scratch/cellxgene_v2_training_v1_scgpt/`

## Resume After Failure

If the script stops or fails, you can resume without reprocessing completed files:

```bash
./scripts/run_remote_extraction.sh --resume
```

This reads `extraction_checkpoint.json` and skips files already completed.

## Troubleshooting

### Check if script is running
```bash
ssh memverge-dataset-curation 'ps aux | grep extract_training'
```

### View console output (including errors)
```bash
ssh memverge-dataset-curation -t 'tail -f /mnt/scratch/logs/extraction_console.log'
```

### Verify training files exist
```bash
ssh memverge-dataset-curation 'ls /mnt/scratch/cellxgene_v2_training_v1/*.parquet | head'
```

### List output files
```bash
ssh memverge-dataset-curation 'ls -lh /mnt/scratch/cellxgene_v2_training_v1_scgpt/'
```

### Stop the script
```bash
ssh memverge-dataset-curation 'tmux kill-session -t scgpt-extraction'
```

## Architecture

The extraction uses DuckDB for efficient large-scale joins:
1. Reads training files from `/mnt/scratch/cellxgene_v2_training_v1/`
2. Extracts dataset UUID from filename (basename without .parquet)
3. Loads corresponding scGPT chunks from `/mnt/scratch/cellxgene_v2_scgpt_chunks/`
4. Performs inner join on `observation_joinid`
5. Writes combined parquet files with ZSTD compression

## Expected Runtime

- Small files (1M rows): ~2-5 minutes per file
- Large files (10M rows): ~15-30 minutes per file
- Total for all files: ~4-12 hours (depends on number and size)

## Files Created

```
/mnt/scratch/cellxgene_v2_training_v1_scgpt/
├── training_001.parquet          # Output files (one per input)
├── training_002.parquet
├── ...
├── extraction_checkpoint.json    # Progress tracking
└── /mnt/scratch/logs/
    ├── extraction.log            # Detailed processing log
    └── extraction_console.log    # Console output (stdout/stderr)
```

## Validation

The script automatically validates:
- Row counts match between input and output
- No null values in critical columns
- File integrity

Check the log for warnings if row counts don't match 100%.
