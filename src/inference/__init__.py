"""
Inference module for cell type classification.

Includes constraint-based post-processing and other inference utilities.
"""

from .constraints import CellxGeneTissueConstraints

__all__ = ['CellxGeneTissueConstraints']
