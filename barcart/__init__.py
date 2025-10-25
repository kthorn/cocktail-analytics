"""
Barcart: Backend code for cocktail analytics.

Provides tools for computing recipe and ingredient similarities using
hierarchical ingredient trees and Earth Mover's Distance.
"""

__version__ = "0.1.0"

from barcart.distance import (
    # Utilities
    build_index_to_id,
    build_ingredient_distance_matrix,
    # Tree building
    build_ingredient_tree,
    # Recipe analysis
    build_recipe_volume_matrix,
    compute_emd,
    emd_matrix,
    # Advanced analytics
    expected_ingredient_match_matrix,
    # Neighborhood analysis
    knn_matrix,
    m_step_blosum,
    neighbor_weight_matrix,
    report_ingredient_neighbors,
    # Distance computations
    weighted_distance,
)
from barcart.registry import IngredientRegistry

__all__ = [
    # Core types
    "IngredientRegistry",
    # Tree building
    "build_ingredient_tree",
    # Distance computations
    "weighted_distance",
    "build_ingredient_distance_matrix",
    # Recipe analysis
    "build_recipe_volume_matrix",
    "compute_emd",
    "emd_matrix",
    # Neighborhood analysis
    "knn_matrix",
    "report_ingredient_neighbors",
    "neighbor_weight_matrix",
    # Advanced analytics
    "expected_ingredient_match_matrix",
    "m_step_blosum",
    # Utilities
    "build_index_to_id",
]
