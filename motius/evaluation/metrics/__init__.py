"""Reusable evaluation metric primitives."""

from .physical import (
    PhysicalMetricsConfig,
    aggregate_physical_metrics,
    compute_physical_metrics,
    physical_metrics_from_motion,
    table_scaled_physical_metrics,
)
from .dance_features import extract_geometric_features, extract_kinetic_features
from .t2m import (
    aggregate_t2m_metrics,
    diversity,
    l2_normalize_embeddings,
    r_precision,
    retrieval_audit,
)

__all__ = [
    "PhysicalMetricsConfig",
    "aggregate_physical_metrics",
    "aggregate_t2m_metrics",
    "compute_physical_metrics",
    "diversity",
    "extract_geometric_features",
    "extract_kinetic_features",
    "l2_normalize_embeddings",
    "physical_metrics_from_motion",
    "r_precision",
    "retrieval_audit",
    "table_scaled_physical_metrics",
]
