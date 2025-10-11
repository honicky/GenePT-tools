#!/usr/bin/env python
"""
Validate generated embeddings from scGPT and Transcriptformer.

This script performs comprehensive validation checks on embedding files.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_parquet_file(file_path: Path) -> Dict:
    """Validate a single Parquet embedding file.
    
    Args:
        file_path: Path to Parquet file
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating {file_path.name}")
    
    try:
        # Load Parquet file
        df = pd.read_parquet(file_path)
        
        # Basic info
        n_cells = len(df)
        
        # Identify embedding columns
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        n_dims = len(embedding_cols)
        
        # Extract embeddings
        embeddings = df[embedding_cols].values
        
        # Validation checks
        validation_results = {
            'file': file_path.name,
            'n_cells': n_cells,
            'embedding_dim': n_dims,
            'has_nan': bool(np.any(np.isnan(embeddings))),
            'has_inf': bool(np.any(np.isinf(embeddings))),
            'min_value': float(np.min(embeddings)),
            'max_value': float(np.max(embeddings)),
            'mean_value': float(np.mean(embeddings)),
            'std_value': float(np.std(embeddings)),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'metadata_columns': [col for col in df.columns if not col.startswith('embedding_')],
            'status': 'valid'
        }
        
        # Check for required metadata
        required_metadata = ['cell_id', 'cell_type']
        missing_metadata = [col for col in required_metadata if col not in df.columns]
        validation_results['missing_metadata'] = missing_metadata
        
        # Check for anomalies
        if validation_results['has_nan']:
            validation_results['status'] = 'invalid - contains NaN'
            logger.error(f"Found NaN values in {file_path.name}")
        
        if validation_results['has_inf']:
            validation_results['status'] = 'invalid - contains Inf'
            logger.error(f"Found Inf values in {file_path.name}")
        
        # Check for zero variance (all same value)
        if validation_results['std_value'] < 1e-10:
            validation_results['status'] = 'warning - zero variance'
            logger.warning(f"Zero variance detected in {file_path.name}")
        
        # Check expected embedding dimensions
        expected_dims = {
            'scgpt': 512,
            'transcriptformer': 256  # Or 128, adjust based on model
        }
        
        for model_name, expected_dim in expected_dims.items():
            if model_name in file_path.name.lower():
                if n_dims != expected_dim:
                    validation_results['status'] = f'warning - unexpected dimension (expected {expected_dim})'
                    logger.warning(f"Unexpected dimension in {file_path.name}: {n_dims} (expected {expected_dim})")
        
        logger.info(f"✓ Validated {file_path.name}: {n_cells} cells, {n_dims} dims, status: {validation_results['status']}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Failed to validate {file_path}: {e}")
        return {
            'file': file_path.name,
            'status': f'error - {str(e)}',
            'error': str(e)
        }


def compare_embeddings(file1: Path, file2: Path) -> Dict:
    """Compare two embedding files for consistency.
    
    Args:
        file1: First embedding file
        file2: Second embedding file
        
    Returns:
        Comparison results
    """
    logger.info(f"Comparing {file1.name} vs {file2.name}")
    
    try:
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)
        
        # Check cell IDs match
        if 'cell_id' in df1.columns and 'cell_id' in df2.columns:
            cells1 = set(df1['cell_id'])
            cells2 = set(df2['cell_id'])
            
            common_cells = cells1 & cells2
            only_in_1 = cells1 - cells2
            only_in_2 = cells2 - cells1
            
            comparison = {
                'file1': file1.name,
                'file2': file2.name,
                'cells_file1': len(cells1),
                'cells_file2': len(cells2),
                'common_cells': len(common_cells),
                'only_in_file1': len(only_in_1),
                'only_in_file2': len(only_in_2),
                'cell_overlap_ratio': len(common_cells) / max(len(cells1), len(cells2))
            }
            
            if len(common_cells) == 0:
                logger.warning("No common cells found between files")
            elif len(only_in_1) > 0 or len(only_in_2) > 0:
                logger.warning(f"Cell mismatch: {len(only_in_1)} unique to file1, {len(only_in_2)} unique to file2")
            else:
                logger.info("✓ All cells match between files")
            
            return comparison
        else:
            return {
                'error': 'Missing cell_id column in one or both files'
            }
            
    except Exception as e:
        return {
            'error': str(e)
        }


def validate_directory(directory: Path, pattern: str = "*.parquet") -> List[Dict]:
    """Validate all embedding files in a directory.
    
    Args:
        directory: Directory containing embedding files
        pattern: File pattern to match
        
    Returns:
        List of validation results
    """
    files = list(directory.glob(pattern))
    logger.info(f"Found {len(files)} files to validate in {directory}")
    
    results = []
    for file_path in sorted(files):
        result = validate_parquet_file(file_path)
        results.append(result)
    
    return results


def generate_summary_report(results: List[Dict]) -> Dict:
    """Generate summary statistics from validation results.
    
    Args:
        results: List of validation results
        
    Returns:
        Summary dictionary
    """
    valid_files = [r for r in results if 'valid' in r.get('status', '')]
    invalid_files = [r for r in results if 'invalid' in r.get('status', '')]
    warning_files = [r for r in results if 'warning' in r.get('status', '')]
    
    total_cells = sum(r.get('n_cells', 0) for r in valid_files)
    total_size_mb = sum(r.get('file_size_mb', 0) for r in results)
    
    summary = {
        'total_files': len(results),
        'valid_files': len(valid_files),
        'invalid_files': len(invalid_files),
        'warning_files': len(warning_files),
        'total_cells': total_cells,
        'total_size_mb': total_size_mb,
        'average_cells_per_file': total_cells / len(valid_files) if valid_files else 0,
        'embedding_dimensions': list(set(r.get('embedding_dim', 0) for r in results if r.get('embedding_dim'))),
        'validation_issues': [
            {
                'file': r.get('file'),
                'status': r.get('status'),
                'error': r.get('error')
            }
            for r in results if 'invalid' in r.get('status', '') or 'error' in r.get('status', '')
        ]
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate embedding files")
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input file or directory to validate'
    )
    parser.add_argument(
        '--compare',
        type=Path,
        help='Second file/directory for comparison'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for validation report'
    )
    parser.add_argument(
        '--pattern',
        default='*.parquet',
        help='File pattern for directory validation'
    )
    
    args = parser.parse_args()
    
    results = []
    
    # Validate single file or directory
    if args.input.is_file():
        result = validate_parquet_file(args.input)
        results = [result]
    elif args.input.is_dir():
        results = validate_directory(args.input, args.pattern)
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return
    
    # Compare if requested
    comparisons = []
    if args.compare:
        if args.input.is_file() and args.compare.is_file():
            comparison = compare_embeddings(args.input, args.compare)
            comparisons = [comparison]
        elif args.input.is_dir() and args.compare.is_dir():
            # Compare matching files
            files1 = sorted(args.input.glob(args.pattern))
            files2 = sorted(args.compare.glob(args.pattern))
            
            # Match by tissue name
            for f1 in files1:
                # Extract tissue name from filename
                tissue = f1.stem.split('_')[1] if '_' in f1.stem else f1.stem
                
                # Find matching file in second directory
                matching = [f2 for f2 in files2 if tissue in f2.name]
                if matching:
                    comparison = compare_embeddings(f1, matching[0])
                    comparisons.append(comparison)
    
    # Generate summary
    summary = generate_summary_report(results)
    
    # Add comparisons to summary
    if comparisons:
        summary['comparisons'] = comparisons
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total files validated: {summary['total_files']}")
    print(f"Valid files: {summary['valid_files']}")
    print(f"Invalid files: {summary['invalid_files']}")
    print(f"Files with warnings: {summary['warning_files']}")
    print(f"Total cells: {summary['total_cells']:,}")
    print(f"Total size: {summary['total_size_mb']:.1f} MB")
    print(f"Embedding dimensions: {summary['embedding_dimensions']}")
    
    if summary['validation_issues']:
        print("\nValidation Issues:")
        for issue in summary['validation_issues']:
            print(f"  - {issue['file']}: {issue['status']}")
    
    if comparisons:
        print("\nComparison Results:")
        for comp in comparisons:
            if 'error' not in comp:
                print(f"  - {comp['file1']} vs {comp['file2']}: ")
                print(f"    Cell overlap: {comp['cell_overlap_ratio']:.1%}")
    
    # Save report if requested
    if args.output:
        # Combine all results
        report = {
            'summary': summary,
            'validation_results': results
        }
        
        if args.output.suffix == '.json':
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif args.output.suffix == '.csv':
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
        else:
            # Default to JSON
            args.output = args.output.with_suffix('.json')
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {args.output}")
    
    # Return code based on validation results
    if summary['invalid_files'] > 0:
        logger.error(f"Validation failed: {summary['invalid_files']} invalid files found")
        exit(1)
    else:
        logger.info("All validations passed!")
        exit(0)


if __name__ == "__main__":
    main()