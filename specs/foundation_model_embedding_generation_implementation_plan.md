# Foundation Model Embedding Generation - Implementation Plan

## Overview
This document provides a detailed, incremental implementation plan for generating scGPT and Transcriptformer embeddings from Tabula Sapiens v2 datasets. Each phase includes concrete deliverables, validation steps, and decision points before proceeding.

## Phase 0: Environment Preparation and Verification ✅
**Duration**: 1 day  
**Goal**: Set up isolated environments and verify basic functionality  
**Status**: COMPLETED

### Tasks

1. **Create separate requirements files** ✅
   - `requirements-scgpt.txt` for Python 3.10 environment
   - `requirements-transcriptformer.txt` for Python 3.11+ environment
   - Deliverable: Two requirements files with model-specific dependencies
   - Validation: Files contain appropriate PyTorch versions and dependencies

2. **Create virtual environments** ✅
   ```bash
   uv venv -p 3.10 .venv-scgpt
   uv venv -p 3.11 .venv-transcriptformer
   ```
   - Deliverable: Two separate virtual environments
   - Validation: Verify Python versions with `python --version` in each env

3. **Install dependencies** ✅
   ```bash
   source .venv-scgpt/bin/activate && uv pip install -r requirements-scgpt.txt
   source .venv-transcriptformer/bin/activate && uv pip install -r requirements-transcriptformer.txt
   ```
   - Deliverable: Fully configured environments
   - Validation: Import test for key packages (torch, scanpy, etc.)

4. **Create helper scripts** ✅
   - `scripts/run_scgpt.sh`
   - `scripts/run_transcriptformer.sh`
   - Deliverable: Executable shell scripts
   - Validation: Test with simple Python command

### Checkpoint 0 ✅
- [x] Can activate each environment independently
- [x] PyTorch versions are correct (2.1.2 for scGPT, 2.8.0 for Transcriptformer)
- [x] MPS is available on macOS (CUDA for Linux)
- [x] Helper scripts execute correctly

**Actual Environment Configuration:**
- **scGPT (.venv-scgpt)**: Python 3.10.0, PyTorch 2.1.2, Transformers 4.34.0, Scanpy 1.10.3
- **Transcriptformer (.venv-transcriptformer)**: Python 3.11.11, PyTorch 2.8.0, Transformers 4.57.0, Scanpy 1.11.4
- **Note**: flash-attn excluded on macOS (requires CUDA)

**Decision Point**: ✅ Environments are properly isolated and functional. Ready to proceed to Phase 1.

---

## Phase 1: Vocabulary Analysis and Coverage Assessment
**Goal**: Understand gene mapping quality before processing

### Tasks

1. **Download model vocabularies**
   ```python
   # Download scGPT vocab.json from HuggingFace or model checkpoint
   # Download Transcriptformer vocabulary
   ```
   - Deliverable: `models/scgpt/vocab.json`, `models/transcriptformer/vocab.json`
   - Validation: JSON files are valid and contain gene mappings

2. **Create vocabulary analysis notebook**
   - File: `notebooks/vocabulary_analysis.ipynb`
   - Implement gene name extraction and cleaning
   - Calculate coverage statistics per tissue
   - Deliverable: Executable notebook with analysis functions
   - Validation: Run on single tissue file first

3. **Run full vocabulary analysis**
   - Process all 26 tissues
   - Generate coverage report
   - Identify missing genes by category
   - Deliverable: `data/vocabulary_analysis_results.csv`
   - Validation: Coverage statistics are reasonable (>0%, <100%)

4. **Create coverage visualization**
   - Bar chart of coverage by tissue
   - Heatmap of missing gene categories
   - Deliverable: `reports/vocabulary_coverage_report.html`
   - Validation: Visualizations are readable and informative

### Checkpoint 1
- [ ] Mean coverage across tissues >85%
- [ ] Minimum coverage for any tissue >75%
- [ ] Key marker genes (CD3E, CD4, CD8A, etc.) are present
- [ ] Missing genes are mostly non-critical (MT-*, pseudogenes)

**Decision Point**: Only proceed if coverage meets minimum thresholds. If not, investigate gene synonym mapping or alternative models.

---

## Phase 2: Single Tissue Test Implementation
**Goal**: Implement and test complete pipeline on smallest dataset

### Tasks

1. **Identify smallest tissue dataset**
   ```python
   # Find tissue with fewest cells for quick iteration
   # Expected: <5000 cells for fast testing
   ```
   - Deliverable: Selected test tissue name
   - Validation: File exists and is readable

2. **Implement scGPT embedding generation**
   - File: `scripts/generate_scgpt_embeddings.py`
   - Core functions:
     - `load_scgpt_model()`
     - `preprocess_adata()`
     - `generate_embeddings_batch()`
     - `validate_embeddings()`
     - `save_to_parquet()`
   - Deliverable: Functional script for single tissue
   - Validation: Script runs without errors

3. **Download scGPT checkpoint**
   ```bash
   # Download from HuggingFace or official source
   # Expected size: ~2-5GB
   ```
   - Deliverable: `models/scgpt/scgpt_human_zero_shot.pt`
   - Validation: Model loads successfully

4. **Test scGPT on single tissue**
   ```bash
   ./scripts/run_scgpt.sh python scripts/generate_scgpt_embeddings.py \
     --tissue [smallest_tissue] \
     --checkpoint-path models/scgpt/scgpt_human_zero_shot.pt
   ```
   - Deliverable: `data/cz_benchmark/embeddings/scgpt/scgpt_[tissue]_embeddings.parquet`
   - Validation: 
     - File size is reasonable (MB not GB for small tissue)
     - No NaN/Inf values
     - Embedding dimension is 512
     - Metadata columns are present

5. **Implement embedding reader and validator**
   - File: `scripts/validate_embeddings.py`
   - Functions to read Parquet and check integrity
   - Deliverable: Validation script
   - Validation: Successfully reads and validates test embedding

### Checkpoint 2
- [ ] Single tissue processed successfully with scGPT
- [ ] Embeddings pass all validation checks
- [ ] Processing time is reasonable (<30 minutes for small tissue)
- [ ] Memory usage stays within limits
- [ ] Parquet file contains both embeddings and metadata

**Decision Point**: Only proceed to full scGPT processing if single tissue works perfectly.

---

## Phase 3: Full scGPT Processing
**Duration**: 2 days  
**Goal**: Generate embeddings for all 26 tissues with scGPT

### Tasks

1. **Add batch processing and checkpointing**
   - Implement resume functionality
   - Add progress tracking
   - Deliverable: Enhanced `generate_scgpt_embeddings.py`
   - Validation: Can interrupt and resume processing

2. **Create processing order by size**
   ```python
   # Sort tissues by cell count (smallest to largest)
   # Process in order to catch issues early
   ```
   - Deliverable: `data/tissue_processing_order.txt`
   - Validation: All 26 tissues are listed

3. **Test on 3 diverse tissues**
   - Small tissue (<10k cells)
   - Medium tissue (10-50k cells)  
   - Large tissue (>50k cells)
   - Deliverable: 3 embedding files
   - Validation: All pass validation checks

4. **Run full scGPT processing**
   ```bash
   ./scripts/run_scgpt.sh python scripts/generate_scgpt_embeddings.py \
     --tissue all \
     --checkpoint-path models/scgpt/scgpt_human_zero_shot.pt \
     --resume
   ```
   - Monitor logs for errors
   - Check progress every few hours
   - Deliverable: 26 scGPT embedding files
   - Validation: Count files, check sizes

5. **Batch validation of all embeddings**
   ```python
   # Run validation script on all generated files
   # Generate summary statistics
   ```
   - Deliverable: `reports/scgpt_validation_summary.csv`
   - Validation: All tissues pass basic checks

### Checkpoint 3
- [ ] All 26 tissues have scGPT embeddings
- [ ] Total size is reasonable (<100GB)
- [ ] No tissues failed processing
- [ ] Validation summary shows no critical issues
- [ ] Processing logs are clean (no repeated errors)

**Decision Point**: Only proceed to Transcriptformer if scGPT is 100% complete.

---

## Phase 4: Transcriptformer Single Tissue Test
**Duration**: 1 day  
**Goal**: Adapt pipeline for Transcriptformer specifics

### Tasks

1. **Implement Transcriptformer embedding generation**
   - File: `scripts/generate_transcriptformer_embeddings.py`
   - Adapt preprocessing for Transcriptformer requirements
   - Deliverable: Functional Transcriptformer script
   - Validation: Script imports run without errors

2. **Download Transcriptformer checkpoint**
   ```bash
   # Download from official repository
   # Expected size: ~1-3GB
   ```
   - Deliverable: `models/transcriptformer/transcriptformer_pretrained.pt`
   - Validation: Model loads successfully

3. **Test on same small tissue**
   ```bash
   ./scripts/run_transcriptformer.sh python scripts/generate_transcriptformer_embeddings.py \
     --tissue [smallest_tissue] \
     --checkpoint-path models/transcriptformer/transcriptformer_pretrained.pt
   ```
   - Deliverable: Test embedding file
   - Validation: Dimension matches expected (128 or 256)

4. **Compare with scGPT results**
   - Check file sizes
   - Compare processing times
   - Validate both can be read
   - Deliverable: Comparison notes
   - Validation: Both formats are consistent

### Checkpoint 4
- [ ] Transcriptformer processes single tissue successfully
- [ ] Embedding dimension is correct
- [ ] Processing time is comparable to scGPT
- [ ] File format matches specification

**Decision Point**: Proceed only if Transcriptformer pipeline is stable.

---

## Phase 5: Full Transcriptformer Processing
**Duration**: 2 days  
**Goal**: Complete embedding generation for all tissues

### Tasks

1. **Run full Transcriptformer processing**
   ```bash
   ./scripts/run_transcriptformer.sh python scripts/generate_transcriptformer_embeddings.py \
     --tissue all \
     --checkpoint-path models/transcriptformer/transcriptformer_pretrained.pt \
     --resume
   ```
   - Use same processing order as scGPT
   - Monitor for consistency
   - Deliverable: 26 Transcriptformer embedding files
   - Validation: File count and sizes

2. **Batch validation**
   - Run same validation as scGPT
   - Compare statistics between models
   - Deliverable: `reports/transcriptformer_validation_summary.csv`
   - Validation: All tissues pass

3. **Create combined summary report**
   - Total files generated
   - Total disk usage
   - Processing times
   - Any failed tissues
   - Deliverable: `reports/embedding_generation_summary.md`
   - Validation: Numbers add up correctly

### Checkpoint 5
- [ ] All 26 tissues have both scGPT and Transcriptformer embeddings
- [ ] Total of 52 Parquet files exist
- [ ] All files are readable and valid
- [ ] Summary report is complete

---

## Phase 6: Final Validation and Packaging
**Duration**: 1 day  
**Goal**: Ensure everything is ready for downstream analysis

### Tasks

1. **Spot check random embeddings**
   - Randomly select 5 tissues
   - Load both scGPT and Transcriptformer embeddings
   - Verify metadata matches between them
   - Deliverable: Validation notebook
   - Validation: Cell IDs align perfectly

2. **Create data manifest**
   ```python
   # Generate JSON/CSV with all file paths, sizes, checksums
   ```
   - Deliverable: `data/cz_benchmark/embeddings/manifest.json`
   - Validation: All files are listed

3. **Generate usage documentation**
   - How to load embeddings
   - Example code snippets
   - Description of metadata fields
   - Deliverable: `data/cz_benchmark/embeddings/README.md`
   - Validation: Code examples work

4. **Archive if needed**
   ```bash
   # Create tar.gz for easy transfer
   tar -czf embeddings.tar.gz data/cz_benchmark/embeddings/
   ```
   - Deliverable: Compressed archive (optional)
   - Validation: Archive extracts correctly

### Final Checkpoint
- [ ] All 52 embedding files exist and are valid
- [ ] Documentation is complete
- [ ] Manifest file is accurate
- [ ] Total disk usage is within expectations (<200GB)
- [ ] Embeddings are ready for downstream analysis

---

## Rollback Plans

### If vocabulary coverage is too low:
1. Investigate gene synonym mapping
2. Try alternative gene name formats
3. Consider different foundation models
4. Document limitations and proceed with available genes

### If single tissue fails:
1. Check GPU memory and reduce batch size
2. Verify model checkpoint integrity
3. Test with even smaller tissue or subsample
4. Debug preprocessing pipeline step by step

### If full processing fails midway:
1. Use checkpoint/resume functionality
2. Process remaining tissues individually
3. Investigate specific tissue that caused failure
4. Consider excluding problematic tissues with documentation

### If validation reveals issues:
1. Re-process affected tissues only
2. Adjust validation thresholds if too strict
3. Document known issues for downstream users
4. Create tissue-specific workarounds

---

## Success Metrics

| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| Tissues processed | 26/26 | 24/26 | <20/26 |
| Gene coverage | >85% mean | >80% mean | <75% mean |
| Validation pass rate | 100% | >95% | <90% |
| Total processing time | <1 week | <2 weeks | >2 weeks |
| Disk usage | <200GB | <300GB | >500GB |
| NaN/Inf values | 0 | 0 | Any |

---

## Timeline Summary

| Phase | Duration | Cumulative | Key Milestone |
|-------|----------|------------|---------------|
| Phase 0 | 1 day | Day 1 | Environments ready |
| Phase 1 | 1 day | Day 2 | Coverage analyzed |
| Phase 2 | 2 days | Day 4 | Single tissue works |
| Phase 3 | 2 days | Day 6 | All scGPT complete |
| Phase 4 | 1 day | Day 7 | Transcriptformer tested |
| Phase 5 | 2 days | Day 9 | All embeddings done |
| Phase 6 | 1 day | Day 10 | Ready for analysis |

**Buffer**: 2-3 additional days for troubleshooting and reprocessing if needed.

---

## Risk Mitigation

### High Risk Areas:
1. **GPU Memory**: Mitigate with dynamic batch size adjustment
2. **Model compatibility**: Test early with single tissue
3. **Disk space**: Monitor usage, compress if needed
4. **Processing time**: Use multiple GPUs if available

### Monitoring:
- Check logs every 2-4 hours during full processing
- Set up disk space alerts
- Monitor GPU utilization
- Track progress in shared document/Slack

---

## Final Deliverables Checklist

- [ ] 26 scGPT embedding files (Parquet format)
- [ ] 26 Transcriptformer embedding files (Parquet format)
- [ ] Vocabulary analysis report
- [ ] Validation summaries for both models
- [ ] Processing logs for debugging
- [ ] Data manifest with checksums
- [ ] README with usage examples
- [ ] Implementation notes and lessons learned

---

## Next Steps After Completion

1. Run downstream classification benchmarks
2. Compare model performance
3. Visualize embeddings with UMAP/t-SNE
4. Share results with team
5. Plan fine-tuning experiments if needed