"""Tests for model aggregation."""

import unittest
import numpy as np
from src.server.aggregator import FedAvgAggregator, FedProxAggregator


class TestFedAvgAggregator(unittest.TestCase):
    """Test cases for FedAvg aggregator."""

    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = FedAvgAggregator()

    def test_aggregate_equal_weights(self):
        """Test aggregation with equal weights."""
        updates = [
            {"w1": np.array([1.0, 2.0]), "w2": np.array([3.0, 4.0])},
            {"w1": np.array([2.0, 3.0]), "w2": np.array([4.0, 5.0])},
        ]
        result = self.aggregator.aggregate(updates)
        self.assertIn("w1", result)
        self.assertIn("w2", result)

    def test_aggregate_weighted(self):
        """Test aggregation with custom weights."""
        updates = [
            {"w1": np.array([1.0, 2.0])},
            {"w1": np.array([3.0, 4.0])},
        ]
        weights = [0.3, 0.7]
        result = self.aggregator.aggregate(updates, weights)
        self.assertIn("w1", result)

    def test_aggregate_empty(self):
        """Test aggregation with empty updates."""
        result = self.aggregator.aggregate([])
        self.assertEqual(result, {})


class TestFedProxAggregator(unittest.TestCase):
    """Test cases for FedProx aggregator."""

    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = FedProxAggregator(mu=0.01)

    def test_fedprox_aggregation(self):
        """Test FedProx aggregation."""
        updates = [
            {"w1": np.array([1.0, 2.0])},
            {"w1": np.array([2.0, 3.0])},
        ]
        result = self.aggregator.aggregate(updates)
        self.assertIn("w1", result)


if __name__ == "__main__":
    unittest.main()
