# Federated Machine Learning for Privacy-Preserving Smart Healthcare Applications

## Overview

This project implements a federated machine learning system that enables multiple healthcare organizations (hospitals, clinics) to collaboratively train machine learning models without sharing sensitive patient data. Using federated learning algorithms such as Federated Averaging (FedAvg), healthcare providers can develop accurate disease prediction and diagnosis systems while preserving patient privacy and complying with data protection regulations.

## Key Features

- **Privacy-Preserving**: Patient data never leaves local institutions
- **Collaborative Learning**: Multiple healthcare organizations train together
- **FedAvg Algorithm**: Aggregates model updates from distributed clients
- **Disease Prediction Models**:
  - Heart Disease Detection
  - Diabetes Prediction
  - Cancer Detection
- **Scalable Architecture**: Support for multiple participating institutions
- **Secure Communication**: Encrypted model parameter exchange
- **Compliance Ready**: HIPAA and GDPR compliant design

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Central Server (Aggregator)              │
│              - Model Aggregation (FedAvg)                   │
│              - Global Model Distribution                    │
│              - Performance Monitoring                       │
└──────────────┬──────────────┬──────────────┬────────────────┘
               │              │              │
       ┌───────▼────┐  ┌──────▼─────┐  ┌───▼──────┐
       │  Hospital  │  │   Clinic   │  │ Research │
       │     A      │  │     B      │  │ Center C │
       │            │  │            │  │          │
       │ Local Model│  │Local Model │  │Local Model│
       │ Training   │  │ Training   │  │ Training │
       │            │  │            │  │          │
       │ Patient    │  │ Patient    │  │ Patient  │
       │ Data       │  │ Data       │  │ Data     │
       └────────────┘  └────────────┘  └──────────┘
```

## Project Structure

```
federated-ml-healthcare/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── .gitignore
├── config/
│   ├── config.yaml
│   └── model_config.json
├── src/
│   ├── __init__.py
│   ├── server/
│   │   ├── __init__.py
│   │   ├── aggregator.py
│   │   ├── server.py
│   │   └── utils.py
│   ├── client/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── local_trainer.py
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── heart_disease_model.py
│   │   ├── diabetes_model.py
│   │   ├── cancer_model.py
│   │   └── base_model.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── fedavg.py
│   │   └── fedprox.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   └── privacy_utils.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       └── encryption.py
├── tests/
│   ├── __init__.py
│   ├── test_aggregator.py
│   ├── test_client.py
│   ├── test_models.py
│   └── test_algorithms.py
├── examples/
│   ├── basic_federated_training.py
│   ├── heart_disease_federation.py
│   ├── diabetes_federation.py
│   └── cancer_federation.py
└── docs/
    ├── ARCHITECTURE.md
    ├── SETUP_GUIDE.md
    ├── API_DOCUMENTATION.md
    └── PRIVACY_CONSIDERATIONS.md
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/reethu-s123/federated-ml-healthcare.git
   cd federated-ml-healthcare
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

## Quick Start

### Running a Basic Federated Training

```python
from src.server.server import FederatedServer
from src.client.client import FederatedClient
from src.algorithms.fedavg import FedAvg

# Initialize server
server = FederatedServer(
    host='localhost',
    port=8080,
    model_type='heart_disease',
    rounds=10
)

# Initialize clients
client1 = FederatedClient(client_id='hospital_a', server_address='localhost:8080')
client2 = FederatedClient(client_id='clinic_b', server_address='localhost:8080')

# Start federated training
server.start_training()
```

## Disease Prediction Models

### Heart Disease Detection
- **Input Features**: Age, sex, cholesterol, blood pressure, etc.
- **Output**: Binary classification (disease/no disease)
- **Dataset**: UCI Heart Disease Dataset

### Diabetes Prediction
- **Input Features**: Glucose, blood pressure, BMI, etc.
- **Output**: Binary classification (diabetic/non-diabetic)
- **Dataset**: PIMA Indian Diabetes Dataset

### Cancer Detection
- **Input Features**: Cell characteristics, texture, perimeter, etc.
- **Output**: Binary classification (malignant/benign)
- **Dataset**: Breast Cancer Wisconsin Dataset

## Federated Learning Algorithms

### Federated Averaging (FedAvg)
The primary algorithm used for global model aggregation:
- Each client trains locally on its own data
- Clients send model updates to the server
- Server averages the updates to create a new global model
- Process repeats for multiple rounds

### Federated Proximal (FedProx) - Optional
- Improved convergence under heterogeneous data
- Better handling of non-IID (non-independent, identically distributed) data

## Privacy & Security Features

- **Differential Privacy**: Noise injection to protect individual records
- **Secure Aggregation**: Encrypted model parameter exchange
- **Data Anonymization**: Patient identifiers removed
- **Access Control**: Role-based permissions for institutions
- **Audit Logging**: Track all model updates and access

## Configuration

Edit `config/config.yaml` to customize:
- Number of federated rounds
- Local training epochs
- Batch size
- Learning rate
- Model architecture
- Privacy parameters (differential privacy epsilon, delta)

## Usage Examples

See the `examples/` directory for complete examples:

1. `basic_federated_training.py` - Simple federated training setup
2. `heart_disease_federation.py` - Heart disease model training
3. `diabetes_federation.py` - Diabetes model training
4. `cancer_federation.py` - Cancer detection model training

## Performance Metrics

The system tracks:
- **Accuracy**: Overall model correctness
- **Precision & Recall**: Per-class performance
- **F1-Score**: Balanced performance metric
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Communication Cost**: Number of rounds and bandwidth usage
- **Privacy Budget**: Differential privacy consumption

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_models.py
pytest tests/test_algorithms.py
```

## Documentation

- **ARCHITECTURE.md** - Detailed system architecture
- **SETUP_GUIDE.md** - Comprehensive setup instructions
- **API_DOCUMENTATION.md** - API reference for all modules
- **PRIVACY_CONSIDERATIONS.md** - Privacy and security guidelines

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{federated_ml_healthcare_2026,
  title={Federated Machine Learning for Privacy-Preserving Smart Healthcare Applications},
  author={Your Name},
  year={2026},
  url={https://github.com/reethu-s123/federated-ml-healthcare}
}
```

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## Acknowledgments

- Inspired by FedAvg paper (McMahan et al., 2016)
- Healthcare datasets from UCI Machine Learning Repository
- Privacy concepts from differential privacy research

---

**Disclaimer**: This project is for educational and research purposes. For production healthcare applications, ensure compliance with HIPAA, GDPR, and other relevant healthcare regulations.
