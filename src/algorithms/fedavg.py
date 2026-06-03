"""Federated Averaging (FedAvg) algorithm implementation."""

import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class FedAvg:
    """Federated Averaging Algorithm.

    Algorithm:
    1. Server initializes global model
    2. In each round:
       a. Server selects subset of clients
       b. Clients download global model
       c. Clients train locally on their data
       d. Clients send model updates to server
       e. Server aggregates updates using weighted average
       f. Server updates global model
    """

    def __init__(self, num_rounds: int = 10, clients_per_round: int = 5):
        """Initialize FedAvg algorithm.

        Args:
            num_rounds: Number of federated learning rounds
            clients_per_round: Number of clients selected per round
        """
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.global_model = None
        self.round_history = []

        logger.info(f"Initialized FedAvg: rounds={num_rounds}, clients_per_round={clients_per_round}")

    def initialize_global_model(self, initial_model: Dict[str, Any]) -> None:
        """Initialize the global model.

        Args:
            initial_model: Initial global model parameters
        """
        self.global_model = initial_model.copy()
        logger.info("Global model initialized")

    def aggregate_client_updates(
        self, client_updates: List[Dict[str, Any]], client_weights: List[float] = None
    ) -> Dict[str, Any]:
        """Aggregate updates from multiple clients.

        Args:
            client_updates: List of client model updates
            client_weights: Optional weights for weighted averaging

        Returns:
            Aggregated global model update
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {}

        num_clients = len(client_updates)
        logger.info(f"Aggregating {num_clients} client updates")

        if client_weights is None:
            # Equal weighting
            client_weights = [1.0 / num_clients] * num_clients
        else:
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # Extract parameters from first update to initialize aggregated model
        aggregated_update = {}
        first_update = client_updates[0]

        for param_name in first_update.keys():
            # Weighted average of parameters
            param_values = [update.get(param_name, 0) for update in client_updates]
            aggregated_update[param_name] = np.average(param_values, weights=client_weights)

        logger.info("Aggregation completed")
        return aggregated_update

    def train_round(self, clients: List[Dict], round_num: int) -> Dict[str, Any]:
        """Execute one round of federated training.

        Args:
            clients: List of client objects
            round_num: Current round number

        Returns:
            Round metrics
        """
        logger.info(f"\n===== FedAvg Round {round_num} =====")

        # Collect client weights (e.g., based on data size)
        client_weights = [client.get("data_size", 1) for client in clients]

        # Simulate collecting client updates
        client_updates = [client.get("model_update", {}) for client in clients]

        # Aggregate updates
        aggregated_update = self.aggregate_client_updates(client_updates, client_weights)

        # Update global model
        if self.global_model is not None and aggregated_update:
            self.global_model.update(aggregated_update)

        # Record round metrics
        round_metrics = {
            "round": round_num,
            "num_clients": len(clients),
            "aggregated_params_count": len(aggregated_update),
        }

        self.round_history.append(round_metrics)
        logger.info(f"Round {round_num} completed: {round_metrics}")

        return round_metrics

    def train(
        self, clients_per_round_data: List[List[Dict]], initial_model: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run complete federated training.

        Args:
            clients_per_round_data: Client data for each round
            initial_model: Initial global model

        Returns:
            Training history
        """
        self.initialize_global_model(initial_model)
        logger.info(f"Starting FedAvg training for {self.num_rounds} rounds")

        for round_num in range(1, self.num_rounds + 1):
            # In practice, select random subset of clients for this round
            clients = clients_per_round_data[0] if clients_per_round_data else []
            self.train_round(clients, round_num)

        logger.info(f"Training completed. Final history: {len(self.round_history)} rounds")
        return self.round_history

    def get_global_model(self) -> Dict[str, Any]:
        """Get the current global model.

        Returns:
            Global model parameters
        """
        return self.global_model.copy() if self.global_model else {}
