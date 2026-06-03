"""Federated Machine Learning for Privacy-Preserving Smart Healthcare Applications."""

__version__ = "0.1.0"
__author__ = "Federated ML Healthcare Contributors"

from src.server.server import FederatedServer
from src.client.client import FederatedClient
from src.algorithms.fedavg import FedAvg

__all__ = [
    "FederatedServer",
    "FederatedClient",
    "FedAvg",
]
