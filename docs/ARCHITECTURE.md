# System Architecture

## Overview

The Federated Machine Learning for Privacy-Preserving Smart Healthcare system consists of three main components:

1. **Central Server** - Aggregates model updates from clients
2. **Client Nodes** - Individual healthcare institutions
3. **Communication Layer** - Secure parameter exchange

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│                  FEDERATED SERVER                        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Model Aggregator (FedAvg/FedProx)                │ │
│  │  - Weighted averaging of model parameters         │ │
│  │  - Convergence checking                           │ │
│  │  - Model versioning                               │ │
│  └────────────────────────────────────────────────────┘ │
│                        ▲                                 │
│                        │ (Global Model Update)           │
│                        │                                 │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Communication Manager                             │ │
│  │  - gRPC Server                                     │ │
│  │  - Encryption/Decryption                          │ │
│  │  - Update validation                              │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘
     │              │              │              │
     │ (Model)      │ (Model)      │ (Model)      │ (Model)
     │              │              │              │
     ▼              ▼              ▼              ▼
  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
  │Hospital│   │ Clinic │   │Research│   │Private │
  │   A    │   │   B    │   │Center C│   │ Clinic │
  └────────┘   └────────┘   └────────┘   └────────┘
      │            │            │            │
  ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
  │Local  │    │Local  │    │Local  │    │Local  │
  │Model  │    │Model  │    │Model  │    │Model  │
  │Train  │    │Train  │    │Train  │    │Train  │
  └───────┘    └───────┘    └───────┘    └───────┘
      │            │            │            │
  ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
  │Patient│    │Patient│    │Patient│    │Patient│
  │Data   │    │Data   │    │Data   │    │Data   │
  └───────┘    └───────┘    └───────┘    └───────┘
```

## Component Details

### 1. Central Server

**Responsibilities:**
- Initialize and maintain global model
- Manage client registration/deregistration
- Select clients for each training round
- Aggregate model updates from clients
- Broadcast updated global model to clients
- Monitor training progress
- Log and audit all operations

**Key Classes:**
- `FederatedServer` - Main server class
- `Aggregator` - Base aggregation class
- `FedAvgAggregator` - FedAvg implementation
- `FedProxAggregator` - FedProx implementation

### 2. Client Nodes

**Responsibilities:**
- Load local healthcare data
- Train local models on private data
- Extract model updates
- Send encrypted updates to server
- Receive updated global model
- Never share raw patient data

**Key Classes:**
- `FederatedClient` - Client interface
- `LocalTrainer` - Local training executor

### 3. Disease Prediction Models

**Supported Models:**
- Heart Disease Detection (13 input features)
- Diabetes Prediction (8 input features)
- Cancer Detection (30 input features)

**Model Architecture:**
- Neural networks with configurable hidden layers
- Binary classification output
- Dropout and regularization support

## Federated Learning Algorithms

### FedAvg (Federated Averaging)

1. Server broadcasts global model to clients
2. Each client trains on local data for E epochs
3. Clients send model updates (w_i) to server
4. Server computes weighted average: w_new = Σ(n_i/n * Δw_i)
5. Server broadcasts new global model
6. Repeat for R rounds

**Parameters:**
- E: Local epochs per round
- R: Total rounds
- n_i: Client i data size
- η: Learning rate

### FedProx (Federated Proximal)

Extends FedAvg with proximal term:

L(w) = f(w) + μ/2 * ||w - w_t||²

**Advantages:**
- Better convergence under non-IID data
- Handles system heterogeneity
- More stable with diverse client datasets

## Privacy & Security

### Privacy Preservation Techniques

1. **Local Training**
   - Data never leaves healthcare institution
   - Only model parameters transmitted

2. **Differential Privacy**
   - Gaussian noise added to updates
   - Privacy budget (ε, δ) tracked

3. **Secure Aggregation**
   - Encrypted model parameters
   - Server cannot view individual updates

4. **Data Anonymization**
   - Patient identifiers removed
   - Feature normalization applied

### Security Considerations

- **Authentication**: mTLS for client-server communication
- **Encryption**: End-to-end encryption of model updates
- **Audit Logging**: All operations logged and timestamped
- **Access Control**: Role-based permissions

## Communication Protocol

**gRPC with TLS:**
- Protocol Buffers for serialization
- Encrypted channels
- Timeout and retry mechanisms
- Load balancing support

**Message Types:**
- Model Request
- Model Update
- Aggregation Status
- Health Check

## Scalability Considerations

1. **Horizontal Scaling**
   - Multiple aggregation servers
   - Load balancing
   - Fault tolerance

2. **Vertical Scaling**
   - Larger model sizes
   - More complex architectures
   - Increased local training capacity

3. **Communication Efficiency**
   - Model compression
   - Gradient quantization
   - Communication scheduling

## Deployment Architecture

```
Production Environment:
┌─────────────────────────────────────┐
│  Load Balancer                      │
│  (Health checks, routing)           │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Server1 │ │Server2 │ │Server3 │
│Master  │ │Replica │ │Replica │
└────────┘ └────────┘ └────────┘
    │          │          │
    └──────────┼──────────┘
               │
          ┌────┴────┐
          │          │
          ▼          ▼
      Database   Logging
       Backup    Service
```

## Error Handling

1. **Client Failures**
   - Automatic retry with exponential backoff
   - Client heartbeat monitoring
   - Replacement with standby client

2. **Network Issues**
   - Timeout handling
   - Partial update recovery
   - Connection pooling

3. **Model Convergence Issues**
   - Learning rate adjustment
   - Early stopping
   - Model checkpointing
