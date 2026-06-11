"""Federated learning example for cancer detection."""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.server.server import FederatedServer
from src.client.client import FederatedClient
from src.models.cancer_model import CancerModel
from src.data.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run cancer detection federated learning example."""
    logger.info("Starting Cancer Detection Federated Learning Example")

    # Initialize server
    server = FederatedServer(
        host="127.0.0.1",
        port=8082,
        model_type="cancer",
        rounds=5,
        clients_per_round=3,
    )

    # Register clients (research centers)
    logger.info("Registering research centers")
    research_centers = [
        ("research_center_a", "192.168.1.60:5005"),
        ("research_center_b", "192.168.1.70:5006"),
        ("research_center_c", "192.168.1.80:5007"),
    ]

    for center_id, center_addr in research_centers:
        server.register_client(center_id, center_addr)

    # Initialize clients
    logger.info("Initializing clients")
    data_loader = DataLoader(test_size=0.2)
    clients = []

    for center_id, center_addr in research_centers:
        client = FederatedClient(center_id, "127.0.0.1:8082")
        client.connect_to_server()
        
        # Generate synthetic data
        X, y = data_loader.generate_synthetic_data(n_samples=150, n_features=30)
        client.load_data(f"data/{center_id}_data.csv", data_size=len(X))
        
        clients.append(client)

    # Start training
    logger.info("Starting federated training")
    server.start_training()

    # Print results
    logger.info("\n=== TRAINING RESULTS ===")
    logger.info(f"Global model: {server.get_global_model()}")
    logger.info(f"Total rounds completed: {len(server.get_training_history())}")

    # Cleanup
    for client in clients:
        client.disconnect_from_server()

    logger.info("Cancer Detection Example completed")


if __name__ == "__main__":
    main()
