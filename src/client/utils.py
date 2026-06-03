"""Utility functions for client operations."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def compute_data_statistics(data_size: int, feature_count: int) -> Dict[str, int]:
    """Compute statistics about client data.

    Args:
        data_size: Size of client dataset
        feature_count: Number of features

    Returns:
        Data statistics
    """
    return {"total_samples": data_size, "total_features": feature_count, "bytes_per_sample": 100}


def validate_model_update(update: Dict) -> bool:
    """Validate model update before sending to server.

    Args:
        update: Model update to validate

    Returns:
        True if update is valid
    """
    required_fields = ["client_id", "model_version", "parameters"]
    return all(field in update for field in required_fields)
