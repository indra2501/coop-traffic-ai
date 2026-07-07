# AutoCar: Cooperative Intelligence for Autonomous Secure Traffic Management

> A decentralized multi-agent traffic management framework integrating **Graph Convolutional Networks (GCN)**, **Deep Q-Networks (DQN)**, **Vehicle-to-Vehicle (V2V) Communication**, and **Federated Reinforcement Learning (FRL)** for intelligent, scalable, and privacy-preserving autonomous traffic coordination.

---

## Overview

Urban traffic environments involve complex interactions among multiple autonomous vehicles, dynamic traffic signals, and continuously changing road conditions. Traditional centralized traffic management approaches suffer from high latency, limited scalability, and single points of failure.

**AutoCar** proposes a decentralized cooperative intelligence framework where autonomous vehicles communicate, learn, and coordinate in real time to improve road safety, traffic efficiency, and scalability.

The framework combines:

- Graph Convolutional Networks (GCN)
- Deep Q-Network (DQN)
- Vehicle-to-Vehicle (V2V) Communication
- Federated Reinforcement Learning (FRL)

to enable collaborative decision-making while preserving local data privacy.

---

# Key Features

- Graph-based vehicle interaction modeling
- Multi-agent reinforcement learning
- Federated policy learning
- Real-time V2V communication
- Predictive collision avoidance
- Dynamic lane changing
- Intelligent speed regulation
- Emergency vehicle prioritization
- Privacy-preserving decentralized learning
- Real-time traffic simulation

---

# System Architecture

```
Vehicle State
      │
      ▼
Feature Extraction
      │
Graph Construction
      │
      ▼
Graph Convolutional Network (GCN)
      │
      ▼
Vehicle Interaction Embeddings
      │
      ▼
Deep Q Network (DQN)
      │
      ▼
Driving Decision
      │
 ┌────┴────┐
 │         │
 ▼         ▼
Vehicle   V2V Communication
Action        │
              ▼
      Local Reinforcement Learning
              │
              ▼
 Federated Model Aggregation (FedAvg)
              │
              ▼
 Updated Global Policy
```

---

# Methodology

The proposed framework consists of the following stages:

## 1. Vehicle State Representation

Each vehicle continuously observes:

- Position
- Velocity
- Heading
- Lane ID
- Traffic signal status
- Nearby vehicles

---

## 2. Graph Construction

Vehicles are represented as graph nodes.

Edges are dynamically formed according to:

- Spatial proximity
- Communication range
- Traffic interaction

The graph is continuously updated as traffic evolves.

---

## 3. Graph Convolutional Network

The GCN captures:

- Neighbor relationships
- Local traffic density
- Interaction-aware features
- Cooperative traffic dynamics

The generated embeddings provide contextual information for decision-making.

---

## 4. Deep Q Network

The DQN predicts the optimal driving action.

Possible actions include:

- Maintain speed
- Accelerate
- Brake
- Emergency brake
- Lane change
- Yield

Reward function optimizes:

- Collision avoidance
- Throughput
- Waiting time
- Traffic stability
- Rule compliance

---

## 5. Vehicle-to-Vehicle Communication

Vehicles exchange:

- Position
- Velocity
- Braking intent
- Hazard alerts
- Emergency messages

Communication improves cooperative decision-making and extends situational awareness.

---

## 6. Federated Reinforcement Learning

Each vehicle maintains a local DQN model.

Instead of sharing raw driving data:

- Local models are trained independently.
- Model parameters are periodically shared.
- Federated Averaging (FedAvg) aggregates local updates.
- A global model is redistributed to participating vehicles.

This preserves privacy while enabling collaborative policy improvement.

---

# Experimental Evaluation

The framework was evaluated using a real-time multi-agent traffic simulation under increasingly complex traffic scenarios.

Performance metrics include:

- Collision avoidance
- Throughput
- Waiting time
- Travel time
- Decision latency
- FPS
- Communication overhead
- Learning convergence

---

# Reported Results

| Metric | Performance |
|----------|------------|
| Collision Avoidance Accuracy | 92% |
| Throughput Improvement | 28% |
| Decision Latency | 85 ms |
| Communication Reduction | 60% |
| Policy Variance Reduction | 41% |
| GCN Update Time | 1.4 ms |
| Federated Aggregation | < 5 ms |
| Real-Time Performance | ~60 FPS |
| Active Vehicles | 40–55 |

---

# Major Contributions

- Unified GCN–DQN framework for decentralized autonomous traffic coordination.

- Dynamic graph representation of vehicle interactions.

- Privacy-preserving Federated Reinforcement Learning.

- Cooperative Vehicle-to-Vehicle communication.

- Predictive collision prevention.

- Adaptive lane-changing and speed regulation.

- Comprehensive evaluation using safety, efficiency, communication, and computational metrics.

---

# Advantages

- Decentralized architecture
- No single point of failure
- Privacy-preserving learning
- Real-time inference
- Scalable to large traffic networks
- Communication-aware decision-making
- Adaptive cooperative driving
- High collision avoidance accuracy

---

# Technologies Used

## Programming

- Python

## Machine Learning

- PyTorch
- Deep Q Networks
- Graph Neural Networks

## Simulation

- Pygame
- NumPy

## Graph Processing

- NetworkX

## Communication

- Vehicle-to-Vehicle Messaging

---

# Repository Structure

```
AutoCar/
│
├── models/
│     ├── gcn.py
│     ├── dqn.py
│     └── federated.py
│
├── vehicles/
│
├── communication/
│
├── simulation/
│
├── traffic/
│
├── visualization/
│
├── utils/
│
├── configs/
│
├── assets/
│
├── main.py
│
├── requirements.txt
│
└── README.md
```

---

# Future Work

- Real-world autonomous vehicle validation
- Integration with CARLA and SUMO
- Large-scale city traffic networks
- Advanced Graph Attention Networks (GAT)
- Multi-Agent PPO
- Secure Aggregation
- Differential Privacy
- Digital Twin-based smart city deployment

---

# Citation

If you use this repository, please cite:

```
A. Nandi, A. K. Das, I. Bhattacharya, et al.

Cooperative Intelligence for Autonomous Secure Traffic Management Using Graph Neural Network and Federated Reinforcement Learning.

2026.
```

---

# License

This project is intended for academic and research purposes.

---

# Authors

- Apurba Nandi
- Avik Kumar Das
- Indrashish Bhattacharya
- Sayan Bhandari
- Soumabha Ghosh
- Sayannya Bhuniya
- Sandip Mandal
- Satyajit Chakraborty

---

# Contact

For research collaboration or implementation details, please contact the corresponding author listed in the associated publication.
