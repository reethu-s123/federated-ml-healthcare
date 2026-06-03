# Setup and Installation Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 2GB for dependencies and models
- **Network**: Internet connection for downloading packages

### Required Software

```bash
# Check Python version
python --version

# Should be 3.8 or higher
```

### Optional but Recommended

- Docker: For containerized deployment
- Git: For version control
- PostgreSQL: For production database
- Redis: For caching

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/reethu-s123/federated-ml-healthcare.git
cd federated-ml-healthcare
```

### 2. Create Virtual Environment

**On Linux/macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Package in Development Mode

```bash
pip install -e .
```

### 6. Verify Installation

```bash
python -c "import src; print(src.__version__)"
```

## Configuration

### Configuration File Setup

1. **Main Configuration**
   - Location: `config/config.yaml`
   - Controls server, client, and model settings
   - Edit to customize parameters

2. **Model Configuration**
   - Location: `config/model_config.json`
   - Defines model architectures
   - Feature specifications

### Setting Environment Variables

```bash
# Create .env file
echo "FLASK_ENV=development" > .env
echo "SERVER_HOST=127.0.0.1" >> .env
echo "SERVER_PORT=8080" >> .env
echo "LOG_LEVEL=INFO" >> .env
```

### Configuring Privacy Parameters

```yaml
# config/config.yaml
privacy:
  enable_differential_privacy: true
  epsilon: 1.0          # Privacy budget
  delta: 0.00001        # Failure probability
  noise_type: "gaussian"
```

## Running the System

### Starting the Server

```bash
# From project root
python examples/basic_federated_training.py
```

### Starting Individual Clients

Create a client startup script:

```python
# client_startup.py
from src.client.client import FederatedClient

client = FederatedClient("hospital_a", "127.0.0.1:8080")
client.connect_to_server()
client.load_data("data/hospital_a_data.csv", data_size=1000)
client.train_local_model(epochs=5)
client.send_model_update()
```

Run with:

```bash
python client_startup.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Docker Deployment

Create Dockerfile:

```dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "examples/basic_federated_training.py"]
```

Build and run:

```bash
docker build -t federated-ml-healthcare .
docker run federated-ml-healthcare
```

## Troubleshooting

### Issue: Module Import Errors

**Solution:**
```bash
# Reinstall package in development mode
pip install -e .

# Or add project root to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/federated-ml-healthcare"
```

### Issue: Port Already in Use

**Solution:**
```bash
# Change port in config.yaml
server:
  port: 8081  # Use different port

# Or kill process using port
lsof -i :8080
kill -9 <PID>
```

### Issue: Out of Memory

**Solution:**
- Reduce batch size in config
- Use fewer hidden layers
- Enable gradient checkpointing

### Issue: Connection Refused

**Solution:**
```bash
# Check if server is running
netstat -an | grep 8080

# Ensure firewall allows connections
# On macOS
sudo pfctl -f /etc/pf.conf

# On Linux
sudo ufw allow 8080
```

### Issue: Low Accuracy

**Solution:**
- Increase number of federated rounds
- Increase local training epochs
- Adjust learning rate
- Check data quality
- Normalize features

## Next Steps

1. Read [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
2. Review [PRIVACY_CONSIDERATIONS.md](PRIVACY_CONSIDERATIONS.md)
3. Explore examples in `examples/` directory
4. Run tests: `pytest tests/`
5. Start with basic federated training
6. Scale to multiple institutions

## Getting Help

- Check documentation in `docs/`
- Review example code in `examples/`
- Check GitHub issues
- Review logs in `logs/`
