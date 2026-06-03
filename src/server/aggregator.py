"""Model aggregation strategies for federated learning."""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Aggregator:
    """Base class for model aggregation."""

    def __init__(self):
        """Initialize aggregator."""
        self.aggregation_history = []

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate updates from clients.

        Args:
            client_updates: List of model updates from clients

        Returns:
            Aggregated global model update
        """
        raise NotImplementedError


class FedAvgAggregator(Aggregator):
    """Federated Averaging aggregator."""

    def aggregate(self, client_updates: List[Dict[str, Any]], weights: List[float] = None) -> Dict[str, Any]:
        """Aggregate model updates using FedAvg algorithm.

        Args:
            client_updates: List of model weights/parameters from clients
            weights: Optional weights for each client (e.g., data size ratios)

        Returns:
            Aggregated global model parameters
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {}

        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        logger.info(f"Aggregating {len(client_updates)} client updates")

        # Initialize aggregated parameters
        aggregated_params = {}

        # Get parameter names from first update
        first_update = client_updates[0]
        param_names = first_update.keys()

        # Weighted average aggregation
        for param_name in param_names:
            param_values = [update[param_name] for update in client_updates]
            aggregated_params[param_name] = np.average(param_values, axis=0, weights=weights)

        self.aggregation_history.append(aggregated_params)
        logger.info("Aggregation completed successfully")

        return aggregated_params


class FedProxAggregator(Aggregator):
    """Federated Proximal aggregator for heterogeneous data."""

    def __init__(self, mu: float = 0.01):
        """Initialize FedProx aggregator.

        Args:
            mu: Proximal term weight
        """
        super().__init__()
        self.mu = mu

    def aggregate(self, client_updates: List[Dict[str, Any]], weights: List[float] = None) -> Dict[str, Any]:
        """Aggregate model updates using FedProx algorithm.

        Args:
            client_updates: List of model updates from clients
            weights: Optional weights for each client

        Returns:
            Aggregated global model parameters
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {}

        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        logger.info(f"Aggregating {len(client_updates)} client updates using FedProx")

        # For FedProx, standard weighted average is used
        # The proximal term is applied during local client training
        aggregated_params = {}
        first_update = client_updates[0]
        param_names = first_update.keys()

        for param_name in param_names:
            param_values = [update[param_name] for update in client_updates]
            aggregated_params[param_name] = np.average(param_values, axis=0, weights=weights)

        self.aggregation_history.append(aggregated_params)
        logger.info("FedProx aggregation completed")

        return aggregated_params
