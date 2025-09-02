# Hierarchical Metrics Implementation Specification

## Overview
Extend the current evaluation metrics in `src/training/metrics.py` to include hierarchical precision, recall, and F1 scores based on the Cell Ontology. This will provide more nuanced evaluation of cell type predictions by considering the ontological relationships between cell types.

## Background
The current evaluation uses flat classification metrics that treat all misclassifications equally. However, cell types exist in a hierarchy (e.g., "T cell" → "CD4-positive T cell" → "regulatory T cell"). A prediction of "CD4-positive T cell" when the truth is "regulatory T cell" should be considered better than predicting "B cell".

## Requirements

### 1. Core Hierarchical Metrics Functions

Add the following functions to `src/training/metrics.py`:

```python
def get_ancestors(node: str, graph: nx.DiGraph) -> Set[str]:
    """Get all ancestors of a node including the node itself."""
    
def calculate_hierarchical_f_score(
    y_true_labels: List[str], 
    y_pred_labels: List[str], 
    ontology_graph: nx.DiGraph
) -> Dict[str, float]:
    """
    Calculate hierarchical precision, recall, and F-score.
    Based on Kiritchenko et al. (2005).
    
    Returns dict with keys:
    - hierarchical_precision
    - hierarchical_recall  
    - hierarchical_f1
    """
```

### 2. Ontology Loading and Caching

Create `src/training/ontology.py`:

```python
class CellOntologyManager:
    """Manages Cell Ontology loading and graph construction."""
    
    def __init__(self, cache_dir: Path):
        """Initialize with cache directory for ontology files."""
        
    def download_ontology(self) -> Path:
        """Download Cell Ontology OBO file if not cached."""
        
    def build_cell_type_graph(self) -> nx.DiGraph:
        """Build directed graph from ontology with caching."""
        
    def map_cell_types_to_ontology(
        self, 
        cell_types: List[str]
    ) -> Dict[str, str]:
        """Map dataset cell types to ontology term IDs."""
```

### 3. Extended Evaluation Function

Update `src/training/metrics.py`:

```python
def evaluate_with_hierarchy(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    cell_types: List[str],
    cell_type_to_idx: Dict[str, int],
    ontology_graph: Optional[nx.DiGraph] = None,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
    k_values: Tuple[int, ...] = (2, 5, 10)
) -> Dict[str, float]:
    """
    Evaluate with both standard and hierarchical metrics.
    
    Returns all standard metrics plus:
    - hierarchical_precision
    - hierarchical_recall
    - hierarchical_f1
    """
```

### 4. Integration with Trainer

Update `src/training/trainer.py`:

1. Add optional ontology support:
```python
class MLPTrainer:
    def __init__(
        self,
        ...,
        enable_hierarchical_metrics: bool = False,
        ontology_cache_dir: Optional[Path] = None
    ):
        if enable_hierarchical_metrics:
            self.ontology_manager = CellOntologyManager(ontology_cache_dir)
            self.ontology_graph = self.ontology_manager.build_cell_type_graph()
```

2. Update validation to use hierarchical metrics when enabled:
```python
def validate(self):
    if self.ontology_graph:
        metrics = evaluate_with_hierarchy(...)
    else:
        metrics = evaluate(...)
```

### 5. WandB Reporting

Ensure hierarchical metrics are logged to WandB:

```python
# In trainer.py validate() method
if self.wandb_run:
    wandb.log({
        **{f"val/{k}": v for k, v in metrics.items()},
        "global_step": self.global_step,
        # Add hierarchical metrics visualization
        "hierarchical_improvement": metrics.get("hierarchical_f1", 0) - metrics.get("macro_f1", 0)
    })
```

### 6. CLI Integration

Update `scripts/train_cellxgene_mlp.py`:

```python
parser.add_argument(
    "--disable-hierarchical-metrics",
    action="store_false",
    help="Disable hierarchical evaluation using Cell Ontology"
)
parser.add_argument(
    "--ontology-cache-dir",
    type=Path,
    default=Path("data/ontology"),
    help="Directory to cache Cell Ontology files"
)
```

### 7. Training Output Enhancement

Update console output during training to show hierarchical metrics:

```
Epoch 1/10 | Step 1000/5000
Loss: 2.456 | LR: 0.0001
Standard Metrics:
  Recall@10: 0.8704 | MRR@10: 0.5605 | DCG@10: 0.6350
  Macro F1: 0.0918 | Precision: 0.1093 | Recall: 0.0924
Hierarchical Metrics:
  H-Precision: 0.9127 | H-Recall: 0.9030 | H-F1: 0.9078
  Improvement over flat: +0.8160
```

## Implementation Notes

### Dependencies
- Add to `pyproject.toml`:
  - `obonet>=1.0.0` for OBO file parsing
  - `networkx>=3.0` for graph operations

### Performance Considerations
1. Cache the ontology graph after first build (pickle or joblib)
2. Pre-compute all ancestor sets for efficiency
3. Consider batched computation for large validation sets

### Testing Requirements
1. Unit tests for hierarchical metric calculation
2. Test with mock ontology graph
3. Integration test with real Cell Ontology subset
4. Verify WandB logging includes new metrics

### Backward Compatibility
- All changes should be backward compatible
- Hierarchical metrics are opt-out via flag
- Existing training scripts continue to work unchanged

## Validation Targets

Based on the demo notebook, expect:
- Hierarchical F1 significantly higher than macro F1 (0.90+ vs 0.09)
- This demonstrates the model learns biologically meaningful relationships
- Large improvement indicates correct ontological structure learning

## Future Extensions

1. **Ontology-aware loss function**: Weight misclassifications by ontological distance
2. **Per-level metrics**: Report accuracy at different ontology depths
3. **Confusion matrix visualization**: Show hierarchical relationships in errors
4. **Unknown cell type detection**: Use ontology to suggest parent types for novel cells

## References

- Kiritchenko, S., Matwin, S., & Famili, A. F. (2005). Functional annotation of genes using hierarchical text categorization.
- Cell Ontology: https://github.com/obophenotype/cell-ontology
- OnClass paper: https://www.nature.com/articles/s41467-021-22961-3