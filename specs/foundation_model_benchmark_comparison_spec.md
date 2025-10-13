# Foundation Model Embedding Comparison Specification

## Executive Summary

This specification describes the implementation strategy for comparing foundation model embeddings (scGPT, Transcriptformer) with GenePT embeddings on the CZ benchmark cell-type classification task. We'll evaluate individual embeddings and their combinations using logistic regression classifiers in a 5-fold cross-validation setup.

## Evaluation Matrix

### Models to Compare
1. **scGPT alone** (512 dimensions)
2. **scGPT + GenePT** (512 + 512 = 1024 dimensions)  
3. **Transcriptformer alone** (2048 dimensions)
4. **Transcriptformer + GenePT** (2048 + 512 = 2560 dimensions)

### Baseline Comparisons (Optional)
5. **GenePT alone** (512 dimensions) - for reference
6. **Original CZ benchmark methods** (if available)

## Architecture Overview

### Embedding Sources
- **scGPT**: Zero-shot embeddings from whole-human model
  - Dimension: 512
  - Location: `data/cz_benchmark/embeddings/scgpt/scgpt_{tissue}_embeddings.parquet`
  - Pre-processing: Gene name cleaning (SYMBOL_ENSG format)
  
- **Transcriptformer**: Zero-shot embeddings 
  - Dimension: 2048
  - Location: `data/cz_benchmark/embeddings/transcriptformer/transcriptformer_{tissue}_embeddings.parquet`
  - Pre-processing: Gene name mapping

- **GenePT**: Pre-computed gene embeddings
  - Dimension: First 512 of 3072 available
  - Location: `data/huggingface_model/embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet`
  - Pre-processing: Expression-weighted averaging, scaling by 0.026

### Classification Strategy
- **Classifier**: Logistic Regression with balanced class weights
- **Regularization**: L2 penalty with C hyperparameter tuning
- **Solver**: LBFGS for small datasets, SAGA for larger ones
- **Multi-class**: One-vs-rest strategy

## Implementation Components

### 1. Data Loading Module
```python
class EmbeddingLoader:
    """Load and align different embedding types."""
    
    def __init__(self, tissue_name: str, embedding_dir: Path):
        self.tissue_name = tissue_name
        self.embedding_dir = embedding_dir
        self.embeddings_cache = {}
    
    def load_scgpt_embeddings(self) -> pd.DataFrame:
        """Load scGPT embeddings for tissue."""
        path = self.embedding_dir / "scgpt" / f"scgpt_{self.tissue_name}_embeddings.parquet"
        df = pd.read_parquet(path)
        
        # Extract cell metadata and embeddings
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        metadata_cols = ['cell_id', 'cell_type', 'tissue', 'donor_id']
        
        return df[metadata_cols + embedding_cols]
    
    def load_transcriptformer_embeddings(self) -> pd.DataFrame:
        """Load Transcriptformer embeddings for tissue."""
        path = self.embedding_dir / "transcriptformer" / f"transcriptformer_{self.tissue_name}_embeddings.parquet"
        return pd.read_parquet(path)
    
    def load_genept_embeddings(self, adata) -> np.ndarray:
        """
        Generate GenePT embeddings for cells using expression-weighted averaging.
        
        Steps:
        1. Load gene embeddings from parquet
        2. Clean gene names from adata (handle SYMBOL_ENSG format)
        3. Align with adata gene names
        4. Create cell embeddings via weighted averaging
        5. Scale by 0.026 for OpenAI embeddings
        """
        # Load gene embeddings
        gene_embeddings_df = pd.read_parquet(
            "data/huggingface_model/embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet"
        )
        
        # Use first 512 dimensions
        embedding_cols = [str(i) for i in range(512)]
        gene_embeddings = gene_embeddings_df[embedding_cols].values
        gene_names = gene_embeddings_df.index.tolist()
        
        # Clean gene names from adata - handle SYMBOL_ENSG format
        # As documented in implementation plan:
        # - Most genes are clean symbols (20,075 out of 26,167)
        # - 252 genes have SYMBOL_ENSG format (e.g., "MATR3_ENSG00000015479")
        # - 5,840 genes are ENSG IDs only
        clean_gene_names = []
        for gene in adata.var['feature_name']:
            if '_ENSG' in gene:
                # Extract symbol from SYMBOL_ENSG format
                clean_gene_names.append(gene.split('_ENSG')[0])
            elif gene.startswith('ENSG'):
                # Keep ENSG IDs as-is (won't match GenePT embeddings)
                clean_gene_names.append(gene)
            else:
                # Already clean symbol
                clean_gene_names.append(gene)
        
        # Create embedding matrix aligned with cleaned adata genes
        from src.inference import create_embedding_matrix, create_cell_embeddings
        embedding_matrix, valid_indices = create_embedding_matrix(
            gene_embeddings_df.reset_index(),
            clean_gene_names,
            id_column='index'
        )
        
        # Generate cell embeddings
        # Note: create_cell_embeddings handles sparse matrices efficiently
        cell_embeddings = create_cell_embeddings(adata.X, embedding_matrix, valid_indices)
        
        # Scale for OpenAI embeddings
        return cell_embeddings / 0.026
    
    def combine_embeddings(self, *embeddings: np.ndarray) -> np.ndarray:
        """Concatenate multiple embedding matrices."""
        return np.hstack(embeddings)
```

### 2. Cross-Validation Framework
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    
    # Data settings
    tissue_names: List[str] = field(default_factory=lambda: [
        'Blood', 'Bone_Marrow', 'Lung', 'Mammary', 'Thymus'
    ])
    embedding_dir: Path = Path("data/cz_benchmark/embeddings")
    benchmark_data_dir: Path = Path("/Users/rj/personal/Tabula_Sapiens_v2_Curated_Benchmark")
    
    # Cross-validation settings
    n_folds: int = 5
    random_state: int = 42
    stratify: bool = True
    
    # Logistic regression hyperparameters
    C_values: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    max_iter: int = 1000
    solver: str = 'lbfgs'  # or 'saga' for large datasets
    class_weight: str = 'balanced'
    n_jobs: int = -1
    
    # WandB settings
    wandb_project: str = "foundation-model-comparison"
    wandb_entity: Optional[str] = None
    track_experiments: bool = True

class CrossValidationEvaluator:
    """Run cross-validation for different embedding combinations."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
    
    def evaluate_embedding_combination(
        self,
        X: np.ndarray,
        y: np.ndarray,
        embedding_name: str,
        tissue_name: str
    ) -> Dict:
        """
        Run 5-fold CV for a specific embedding combination.
        
        Returns:
            Dictionary with per-fold and aggregated metrics
        """
        cv = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        fold_results = []
        best_C_per_fold = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Hyperparameter tuning on validation split
            best_C, best_score = self.tune_hyperparameters(
                X_train, y_train, fold_idx
            )
            best_C_per_fold.append(best_C)
            
            # Train final model with best C
            clf = LogisticRegression(
                C=best_C,
                max_iter=self.config.max_iter,
                solver=self.config.solver,
                class_weight=self.config.class_weight,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state
            )
            
            clf.fit(X_train, y_train)
            
            # Evaluate on test fold
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            
            metrics = self.compute_metrics(y_test, y_pred, y_prob)
            metrics['fold'] = fold_idx
            metrics['best_C'] = best_C
            fold_results.append(metrics)
            
            # Log to WandB if enabled
            if self.config.track_experiments:
                wandb.log({
                    f"{embedding_name}_{tissue_name}_fold_{fold_idx}_accuracy": metrics['accuracy'],
                    f"{embedding_name}_{tissue_name}_fold_{fold_idx}_f1_macro": metrics['f1_macro']
                })
        
        # Aggregate results
        aggregated = self.aggregate_metrics(fold_results)
        aggregated['embedding_name'] = embedding_name
        aggregated['tissue_name'] = tissue_name
        aggregated['best_C_values'] = best_C_per_fold
        aggregated['n_features'] = X.shape[1]
        aggregated['n_samples'] = X.shape[0]
        aggregated['n_classes'] = len(np.unique(y))
        
        return aggregated
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fold_idx: int
    ) -> Tuple[float, float]:
        """
        Tune C parameter using validation split.
        
        Returns:
            Best C value and validation score
        """
        # Create validation split from training data
        val_size = int(0.2 * len(X_train))
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_tr = X_train[train_indices]
        X_val = X_train[val_indices]
        y_tr = y_train[train_indices]
        y_val = y_train[val_indices]
        
        best_score = -np.inf
        best_C = self.config.C_values[0]
        
        for C in self.config.C_values:
            clf = LogisticRegression(
                C=C,
                max_iter=self.config.max_iter,
                solver=self.config.solver,
                class_weight=self.config.class_weight,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state
            )
            
            clf.fit(X_tr, y_tr)
            score = clf.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                best_C = C
        
        return best_C, best_score
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """Compute classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
    
    def aggregate_metrics(self, fold_results: List[Dict]) -> Dict:
        """Aggregate metrics across folds."""
        metrics = {}
        
        for key in ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']:
            values = [r[key] for r in fold_results]
            metrics[f"{key}_mean"] = np.mean(values)
            metrics[f"{key}_std"] = np.std(values)
            metrics[f"{key}_per_fold"] = values
        
        return metrics
```

### 3. Main Experiment Pipeline
```python
def run_foundation_model_comparison():
    """
    Main pipeline for comparing foundation model embeddings.
    
    Evaluates:
    1. scGPT alone
    2. scGPT + GenePT
    3. Transcriptformer alone
    4. Transcriptformer + GenePT
    """
    import scanpy as sc
    
    config = BenchmarkConfig()
    evaluator = CrossValidationEvaluator(config)
    
    # Initialize WandB
    if config.track_experiments:
        wandb.init(
            project=config.wandb_project,
            name="foundation_model_comparison",
            config=config.__dict__
        )
    
    all_results = []
    
    for tissue_name in config.tissue_names:
        print(f"\n{'='*60}")
        print(f"Processing tissue: {tissue_name}")
        print('='*60)
        
        # Load original data for GenePT embedding generation
        # Note: Files are named homo_sapiens_{uuid}_{tissue}_v2_curated.h5ad
        # where uuid is 10df7690-6d10-4029-a47e-0f071bb2df83
        adata_path = list(config.benchmark_data_dir.glob(
            f"homo_sapiens_*_{tissue_name}_v2_curated.h5ad"
        ))[0]
        adata = sc.read_h5ad(adata_path)
        
        # Handle layers - Tabula Sapiens v2 stores data in X_original layer
        if adata.X is None or (hasattr(adata.X, 'nnz') and adata.X.nnz == 0):
            adata.X = adata.layers['X_original'].copy()
        
        # IMPORTANT: Gene names are in adata.var['feature_name'], not var_names
        # Gene name formats in Tabula Sapiens v2:
        # - Most are clean symbols (20,075 out of 26,167)
        # - 252 genes have SYMBOL_ENSG format (e.g., "MATR3_ENSG00000015479")
        # - 5,840 are ENSG IDs only
        # The load_genept_embeddings function handles this cleaning
        
        # Initialize loader
        loader = EmbeddingLoader(tissue_name, config.embedding_dir)
        
        # Load embeddings
        print("Loading embeddings...")
        scgpt_df = loader.load_scgpt_embeddings()
        transcriptformer_df = loader.load_transcriptformer_embeddings()
        
        # Ensure cell alignment
        # Assuming cell_id matches between files
        cell_ids = scgpt_df['cell_id'].values
        
        # Extract embeddings as numpy arrays
        scgpt_embeddings = scgpt_df[[col for col in scgpt_df.columns 
                                     if col.startswith('embedding_')]].values
        
        transcriptformer_embeddings = transcriptformer_df[[col for col in transcriptformer_df.columns 
                                                           if col.startswith('embedding_')]].values
        
        # Generate GenePT embeddings
        print("Generating GenePT embeddings...")
        genept_embeddings = loader.load_genept_embeddings(adata)
        
        # Get labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(scgpt_df['cell_type'].values)
        
        # Store cell types for reporting
        cell_types = label_encoder.classes_
        
        print(f"Data shape: {len(y)} cells, {len(cell_types)} cell types")
        
        # Experiment 1: scGPT alone
        print("\n1. Evaluating scGPT alone...")
        results_scgpt = evaluator.evaluate_embedding_combination(
            scgpt_embeddings, y, "scgpt", tissue_name
        )
        results_scgpt['cell_types'] = cell_types.tolist()
        all_results.append(results_scgpt)
        
        # Experiment 2: scGPT + GenePT
        print("\n2. Evaluating scGPT + GenePT...")
        combined_scgpt_genept = np.hstack([scgpt_embeddings, genept_embeddings])
        results_scgpt_genept = evaluator.evaluate_embedding_combination(
            combined_scgpt_genept, y, "scgpt_genept", tissue_name
        )
        results_scgpt_genept['cell_types'] = cell_types.tolist()
        all_results.append(results_scgpt_genept)
        
        # Experiment 3: Transcriptformer alone
        print("\n3. Evaluating Transcriptformer alone...")
        results_tf = evaluator.evaluate_embedding_combination(
            transcriptformer_embeddings, y, "transcriptformer", tissue_name
        )
        results_tf['cell_types'] = cell_types.tolist()
        all_results.append(results_tf)
        
        # Experiment 4: Transcriptformer + GenePT
        print("\n4. Evaluating Transcriptformer + GenePT...")
        combined_tf_genept = np.hstack([transcriptformer_embeddings, genept_embeddings])
        results_tf_genept = evaluator.evaluate_embedding_combination(
            combined_tf_genept, y, "transcriptformer_genept", tissue_name
        )
        results_tf_genept['cell_types'] = cell_types.tolist()
        all_results.append(results_tf_genept)
        
        # Experiment 5 (Optional): GenePT alone for reference
        print("\n5. Evaluating GenePT alone (baseline)...")
        results_genept = evaluator.evaluate_embedding_combination(
            genept_embeddings, y, "genept", tissue_name
        )
        results_genept['cell_types'] = cell_types.tolist()
        all_results.append(results_genept)
        
        # Print tissue summary
        print_tissue_summary(tissue_name, all_results[-5:])
    
    # Generate final report
    final_report = generate_comparison_report(all_results)
    
    # Save results
    save_results(all_results, final_report)
    
    if config.track_experiments:
        # Log final summary to WandB
        log_final_summary_to_wandb(final_report)
        wandb.finish()
    
    return all_results, final_report
```

### 4. Result Analysis and Reporting
```python
def print_tissue_summary(tissue_name: str, results: List[Dict]):
    """Print comparison table for a tissue."""
    print(f"\n{tissue_name} Results Summary:")
    print("-" * 80)
    print(f"{'Method':<30} {'Accuracy':<15} {'F1 (Macro)':<15} {'# Features':<10}")
    print("-" * 80)
    
    for r in results:
        method = r['embedding_name']
        acc_mean = r['accuracy_mean']
        acc_std = r['accuracy_std']
        f1_mean = r['f1_macro_mean']
        f1_std = r['f1_macro_std']
        n_features = r['n_features']
        
        print(f"{method:<30} {acc_mean:.3f}±{acc_std:.3f}    "
              f"{f1_mean:.3f}±{f1_std:.3f}    {n_features:<10}")

def generate_comparison_report(all_results: List[Dict]) -> Dict:
    """Generate comprehensive comparison report."""
    import pandas as pd
    from datetime import datetime
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    
    # Aggregate by method across tissues
    method_summary = df.groupby('embedding_name').agg({
        'accuracy_mean': ['mean', 'std'],
        'f1_macro_mean': ['mean', 'std'],
        'n_features': 'first'
    }).round(4)
    
    # Find best method per tissue
    best_per_tissue = df.loc[df.groupby('tissue_name')['accuracy_mean'].idxmax()]
    
    # Statistical comparisons
    comparisons = {}
    
    # Compare combined vs individual embeddings
    for base_method in ['scgpt', 'transcriptformer']:
        base_results = df[df['embedding_name'] == base_method]['accuracy_mean'].values
        combined_results = df[df['embedding_name'] == f"{base_method}_genept"]['accuracy_mean'].values
        
        if len(base_results) > 0 and len(combined_results) > 0:
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(base_results, combined_results)
            
            comparisons[f"{base_method}_vs_combined"] = {
                'base_mean': np.mean(base_results),
                'combined_mean': np.mean(combined_results),
                'improvement': np.mean(combined_results) - np.mean(base_results),
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'n_tissues': len(df['tissue_name'].unique()),
        'n_experiments': len(all_results),
        'method_summary': method_summary.to_dict(),
        'best_per_tissue': best_per_tissue[['tissue_name', 'embedding_name', 
                                           'accuracy_mean', 'f1_macro_mean']].to_dict('records'),
        'statistical_comparisons': comparisons,
        'overall_best': {
            'method': method_summary['accuracy_mean']['mean'].idxmax(),
            'accuracy': method_summary['accuracy_mean']['mean'].max(),
            'f1_macro': method_summary.loc[method_summary['accuracy_mean']['mean'].idxmax(), 
                                          ('f1_macro_mean', 'mean')]
        }
    }
    
    # Print report
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT")
    print("="*80)
    
    print("\nOverall Performance by Method:")
    print(method_summary)
    
    print("\nBest Method per Tissue:")
    for item in report['best_per_tissue']:
        print(f"  {item['tissue_name']:<15} -> {item['embedding_name']:<25} "
              f"(Acc: {item['accuracy_mean']:.3f}, F1: {item['f1_macro_mean']:.3f})")
    
    print("\nStatistical Comparisons (Combined vs Individual):")
    for comparison_name, stats in comparisons.items():
        print(f"\n  {comparison_name}:")
        print(f"    Improvement: {stats['improvement']:.3f}")
        print(f"    P-value: {stats['p_value']:.4f}")
        print(f"    Significant: {stats['significant']}")
    
    print(f"\nOverall Best Method: {report['overall_best']['method']}")
    print(f"  Mean Accuracy: {report['overall_best']['accuracy']:.3f}")
    print(f"  Mean F1 Macro: {report['overall_best']['f1_macro']:.3f}")
    
    return report

def save_results(all_results: List[Dict], report: Dict):
    """Save results to disk."""
    import json
    import pandas as pd
    from pathlib import Path
    
    # Create results directory
    results_dir = Path("results/foundation_model_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    with open(results_dir / "detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save summary report
    with open(results_dir / "comparison_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save as CSV for easy analysis
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / "results_table.csv", index=False)
    
    # Create summary CSV
    summary_data = []
    for r in all_results:
        summary_data.append({
            'tissue': r['tissue_name'],
            'method': r['embedding_name'],
            'accuracy_mean': r['accuracy_mean'],
            'accuracy_std': r['accuracy_std'],
            'f1_macro_mean': r['f1_macro_mean'],
            'f1_macro_std': r['f1_macro_std'],
            'n_features': r['n_features'],
            'n_samples': r['n_samples'],
            'n_classes': r['n_classes']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "summary_table.csv", index=False)
    
    print(f"\nResults saved to {results_dir}")

def log_final_summary_to_wandb(report: Dict):
    """Log final summary metrics to WandB."""
    import wandb
    
    # Create summary table
    summary_table = wandb.Table(
        columns=["Tissue", "Method", "Accuracy", "F1 Macro"],
        data=[[r['tissue_name'], r['embedding_name'], 
               r['accuracy_mean'], r['f1_macro_mean']] 
              for r in report['best_per_tissue']]
    )
    
    wandb.log({
        "best_methods_per_tissue": summary_table,
        "overall_best_method": report['overall_best']['method'],
        "overall_best_accuracy": report['overall_best']['accuracy'],
        "overall_best_f1": report['overall_best']['f1_macro']
    })
    
    # Log statistical comparisons
    for comparison_name, stats in report['statistical_comparisons'].items():
        wandb.log({
            f"comparison_{comparison_name}_improvement": stats['improvement'],
            f"comparison_{comparison_name}_pvalue": stats['p_value'],
            f"comparison_{comparison_name}_significant": stats['significant']
        })
```

### 5. Visualization Module
```python
def create_comparison_plots(all_results: List[Dict], save_dir: Path):
    """Create visualization plots for results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    df = pd.DataFrame(all_results)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Performance by method across tissues
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Accuracy plot
    pivot_acc = df.pivot(index='tissue_name', columns='embedding_name', values='accuracy_mean')
    pivot_acc.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Accuracy by Method and Tissue')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    
    # F1 Macro plot
    pivot_f1 = df.pivot(index='tissue_name', columns='embedding_name', values='f1_macro_mean')
    pivot_f1.plot(kind='bar', ax=axes[1])
    axes[1].set_title('F1 Macro by Method and Tissue')
    axes[1].set_ylabel('F1 Macro Score')
    axes[1].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_by_tissue.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Improvement from combining embeddings
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = []
    for tissue in df['tissue_name'].unique():
        tissue_df = df[df['tissue_name'] == tissue]
        
        for base in ['scgpt', 'transcriptformer']:
            base_acc = tissue_df[tissue_df['embedding_name'] == base]['accuracy_mean'].values
            combined_acc = tissue_df[tissue_df['embedding_name'] == f"{base}_genept"]['accuracy_mean'].values
            
            if len(base_acc) > 0 and len(combined_acc) > 0:
                improvements.append({
                    'tissue': tissue,
                    'base_method': base,
                    'improvement': combined_acc[0] - base_acc[0]
                })
    
    imp_df = pd.DataFrame(improvements)
    pivot_imp = imp_df.pivot(index='tissue', columns='base_method', values='improvement')
    pivot_imp.plot(kind='bar', ax=ax)
    ax.set_title('Accuracy Improvement from Adding GenePT Embeddings')
    ax.set_ylabel('Accuracy Improvement')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Base Method')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'improvement_from_combination.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Feature dimension vs performance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter_data = df[['embedding_name', 'n_features', 'accuracy_mean']].drop_duplicates()
    colors = {'scgpt': 'blue', 'transcriptformer': 'green', 
              'scgpt_genept': 'red', 'transcriptformer_genept': 'orange',
              'genept': 'purple'}
    
    for method in scatter_data['embedding_name'].unique():
        method_data = scatter_data[scatter_data['embedding_name'] == method]
        ax.scatter(method_data['n_features'], method_data['accuracy_mean'], 
                  label=method, color=colors.get(method, 'gray'), s=100, alpha=0.7)
    
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Feature Dimensionality vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'features_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Execution Plan

### Phase 1: Data Preparation
1. Verify all embedding files exist for each tissue
2. Load and align cell IDs across different embedding sources
3. Generate GenePT embeddings for all tissues
4. Verify data integrity and dimensions

### Phase 2: Individual Model Evaluation
1. Run 5-fold CV for scGPT embeddings
2. Run 5-fold CV for Transcriptformer embeddings  
3. Run 5-fold CV for GenePT embeddings (baseline)
4. Track metrics and hyperparameters

### Phase 3: Combined Model Evaluation
1. Run 5-fold CV for scGPT + GenePT
2. Run 5-fold CV for Transcriptformer + GenePT
3. Compare improvements from combinations

### Phase 4: Analysis and Reporting
1. Statistical significance testing
2. Generate comparison plots
3. Create summary report
4. Save all results and artifacts

## Expected Outcomes

### Performance Expectations
- **Individual Models**: Establish baseline performance for each embedding type
- **Combined Models**: Expected 2-5% improvement from combining complementary embeddings
- **Best Performer**: Likely scGPT + GenePT due to higher dimensionality and complementary information

### Key Metrics to Track
- Mean accuracy across tissues
- F1 macro score for handling class imbalance
- Standard deviation across folds
- Statistical significance of improvements
- Computational efficiency (training time)

## Dependencies

### Python Packages
- `scikit-learn >= 1.3.0` (LogisticRegression, metrics)
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `scipy >= 1.10.0` (statistical tests)
- `matplotlib >= 3.7.0` (visualization)
- `seaborn >= 0.12.0` (visualization)
- `wandb >= 0.15.0` (optional, for tracking)

### Data Requirements
- **Embedding Files**: 
  - scGPT: `data/cz_benchmark/embeddings/scgpt/scgpt_{tissue}_embeddings.parquet`
  - Transcriptformer: `data/cz_benchmark/embeddings/transcriptformer/transcriptformer_{tissue}_embeddings.parquet`
- **GenePT Embeddings**: `data/huggingface_model/embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet`
- **Original Data**: `/Users/rj/personal/Tabula_Sapiens_v2_Curated_Benchmark/*.h5ad`

### Hardware Requirements
- RAM: ~16GB for loading all embeddings
- Storage: ~5GB for results and intermediate files
- CPU: Multi-core recommended for parallel CV folds

## Usage Example

```python
# Run the complete comparison pipeline
from pathlib import Path

# Configure experiment
config = BenchmarkConfig(
    tissue_names=['Blood', 'Bone_Marrow', 'Lung', 'Mammary', 'Thymus'],
    C_values=[0.001, 0.01, 0.1, 1.0, 10.0],
    track_experiments=True
)

# Run comparison
results, report = run_foundation_model_comparison()

# Create visualizations
save_dir = Path("results/foundation_model_comparison")
create_comparison_plots(results, save_dir)

# Print final summary
print(f"\nBest overall method: {report['overall_best']['method']}")
print(f"Mean accuracy: {report['overall_best']['accuracy']:.3f}")
```