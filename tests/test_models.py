"""Tests for disease prediction models."""

import unittest
import numpy as np
from src.models.heart_disease_model import HeartDiseaseModel
from src.models.diabetes_model import DiabetesModel
from src.models.cancer_model import CancerModel


class TestHeartDiseaseModel(unittest.TestCase):
    """Test cases for heart disease model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = HeartDiseaseModel()
        self.model.build()

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "HeartDiseaseModel")
        self.assertEqual(self.model.input_features, 13)

    def test_model_training(self):
        """Test model training."""
        X_train = np.random.randn(100, 13)
        y_train = np.random.randint(0, 2, 100)
        history = self.model.train(X_train, y_train, epochs=5)
        self.assertTrue(self.model.is_trained)
        self.assertIn("accuracy", history)

    def test_model_prediction(self):
        """Test model prediction."""
        X_train = np.random.randn(100, 13)
        y_train = np.random.randint(0, 2, 100)
        self.model.train(X_train, y_train, epochs=5)

        X_test = np.random.randn(20, 13)
        predictions = self.model.predict(X_test)
        self.assertEqual(len(predictions), 20)


class TestDiabetesModel(unittest.TestCase):
    """Test cases for diabetes model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DiabetesModel()
        self.model.build()

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "DiabetesModel")
        self.assertEqual(self.model.input_features, 8)


class TestCancerModel(unittest.TestCase):
    """Test cases for cancer model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = CancerModel()
        self.model.build()

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "CancerModel")
        self.assertEqual(self.model.input_features, 30)


if __name__ == "__main__":
    unittest.main()
