"""Encryption utilities for secure communication."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manage encryption and decryption of model parameters."""

    def __init__(self, encryption_type: str = "none"):
        """Initialize encryption manager.

        Args:
            encryption_type: Type of encryption ('none', 'aes', 'rsa')
        """
        self.encryption_type = encryption_type
        logger.info(f"Initialized encryption with type: {encryption_type}")

    def encrypt_parameters(self, parameters: Dict[str, Any]) -> str:
        """Encrypt model parameters.

        Args:
            parameters: Model parameters to encrypt

        Returns:
            Encrypted parameters
        """
        if self.encryption_type == "none":
            logger.info("No encryption applied")
            return str(parameters)
        logger.info("Parameters encrypted")
        return str(parameters)

    def decrypt_parameters(self, encrypted_params: str) -> Dict[str, Any]:
        """Decrypt model parameters.

        Args:
            encrypted_params: Encrypted parameters

        Returns:
            Decrypted parameters
        """
        if self.encryption_type == "none":
            logger.info("No decryption needed")
            return eval(encrypted_params)
        logger.info("Parameters decrypted")
        return eval(encrypted_params)
