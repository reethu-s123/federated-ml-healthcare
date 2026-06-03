"""Basic federated learning training example."""

import logging
from src.server.server import FederatedServer
from src.client.client import FederatedClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run basic federated training example."""
    logger.info("Starting basic federated learning example")

    # Initialize server
    server = FederatedServer(
        host="127.0.0.1", port=8080, model_type="heart_disease", rounds=3, clients_per_round=2
    )

    # Register clients
    logger.info("Registering clients")
    server.register_client("hospital_a", "192.168.1.10:5000")
    server.register_client("clinic_b", "192.168.1.20:5001")
    server.register_client("research_center_c", "192.168.1.30:5002")

    # Initialize clients
    logger.info("Initializing clients")
    client1 = FederatedClient("hospital_a", "127.0.0.1:8080")
    client2 = FederatedClient("clinic_b", "127.0.0.1:8080")

    # Connect clients
    client1.connect_to_server()
    client2.connect_to_server()

    # Start federated training
    logger.info("Starting federated training")
    server.start_training()

    # Print results
    logger.info("Training completed")
    logger.info(f"Training history: {server.get_training_history()}")
    logger.info(f"Global model: {server.get_global_model()}")

    # Disconnect clients
    client1.disconnect_from_server()
    client2.disconnect_from_server()

    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
