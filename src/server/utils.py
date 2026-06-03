"""Utility functions for server operations."""

import logging
import json
from typing import Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


def serialize_model(model: Any) -> str:
    """Serialize model to JSON string.

    Args:
        model: Model object to serialize

    Returns:
        JSON string representation
    """
    try:
        return json.dumps(model, default=str)
    except Exception as e:
        logger.error(f"Error serializing model: {e}")
        return ""


def deserialize_model(model_str: str) -> Dict[str, Any]:
    """Deserialize model from JSON string.

    Args:
        model_str: JSON string representation

    Returns:
        Deserialized model
    """
    try:
        return json.loads(model_str)
    except Exception as e:
        logger.error(f"Error deserializing model: {e}")
        return {}


def calculate_client_weights(client_data_sizes: Dict[str, int]) -> Dict[str, float]:
    """Calculate weights for clients based on data size.

    Args:
        client_data_sizes: Dictionary of client IDs to data sizes

    Returns:
        Dictionary of client IDs to weights
    """
    total_size = sum(client_data_sizes.values())
    if total_size == 0:
        logger.warning("Total data size is 0")
        return {}

    weights = {}
    for client_id, size in client_data_sizes.items():
        weights[client_id] = size / total_size

    return weights


def log_round_metrics(round_num: int, metrics: Dict[str, Any]) -> None:
    """Log round metrics.

    Args:
        round_num: Round number
        metrics: Metrics dictionary
    """
    logger.info(f"\nRound {round_num} Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
