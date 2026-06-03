"""Federated Proximal (FedProx) algorithm implementation."""

import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class FedProx:
    """Federated Proximal (FedProx) Algorithm.

    Improves upon FedAvg by handling non-IID (non-independent, identically distributed) data.
    Adds a proximal term to the local client loss function.
    """

    def __init__(self, num_rounds: int = 10, clients_per_round: int = 5, mu: float = 0.01):
        """Initialize FedProx algorithm.

        Args:
            num_rounds: Number of federated learning rounds
            clients_per_round: Number of clients per round
            mu: Proximal term weight parameter
        """
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.mu = mu
        self.global_model = None
        self.round_history = []

        logger.info(f"Initialized FedProx: rounds={num_rounds}, mu={mu}")

    def initialize_global_model(self, initial_model: Dict[str, Any]) -> None:
        """Initialize the global model.

        Args:
            initial_model: Initial global model parameters
        """
        self.global_model = initial_model.copy()
        logger.info("Global model initialized for FedProx")

    def aggregate_client_updates(
        self, client_updates: List[Dict[str, Any]], client_weights: List[float] = None
    ) -> Dict[str, Any]:
        """Aggregate updates from clients using FedProx strategy.

        Args:
            client_updates: List of client model updates
            client_weights: Optional weights for aggregation

        Returns:
            Aggregated global update
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {}

        num_clients = len(client_updates)
        logger.info(f"FedProx aggregating {num_clients} client updates (mu={self.mu})")

        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        else:
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # Weighted average aggregation (same as FedAvg)
        # Proximal term is applied during local training, not in aggregation
        aggregated_update = {}
        first_update = client_updates[0]

        for param_name in first_update.keys():
            param_values = [update.get(param_name, 0) for update in client_updates]
            aggregated_update[param_name] = np.average(param_values, weights=client_weights)

        logger.info("FedProx aggregation completed")
        return aggregated_update

    def train_round(self, clients: List[Dict], round_num: int) -> Dict[str, Any]:
        """Execute one round of FedProx training.

        Args:
            clients: List of client objects
            round_num: Current round number

        Returns:
            Round metrics
        """
        logger.info(f"\n===== FedProx Round {round_num} (mu={self.mu}) =====")

        client_weights = [client.get("data_size", 1) for client in clients]
        client_updates = [client.get("model_update", {}) for client in clients]

        aggregated_update = self.aggregate_client_updates(client_updates, client_weights)

        if self.global_model is not None and aggregated_update:
            self.global_model.update(aggregated_update)

        round_metrics = {"round": round_num, "num_clients": len(clients), "mu": self.mu}

        self.round_history.append(round_metrics)
        logger.info(f"FedProx Round {round_num} completed")

        return round_metrics

    def get_global_model(self) -> Dict[str, Any]:
        """Get the current global model.

        Returns:
            Global model parameters
        """
        return self.global_model.copy() if self.global_model else {}
