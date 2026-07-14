"""
Hierarchical matrix (H-matrix) structures for efficient matrix approximation and operations.

This package provides:
- ClusterTree: Hierarchical spatial clustering of indices
- BlockClusterTree: Block-structured partition based on admissibility
- HMatrix: H-matrix representation with low-rank and dense blocks
"""

from .cluster_tree import Cluster, ClusterTree
from .block_cluster_tree import Block, BlockClusterTree
from .hmatrix import HMatrix
from .entry_source import IndexedEntrySource, MatrixEntrySource
from . import low_rank_approx

__all__ = [
    "Cluster",
    "ClusterTree",
    "Block",
    "BlockClusterTree",
    "HMatrix",
    "MatrixEntrySource",
    "IndexedEntrySource",
    "low_rank_approx",
]
