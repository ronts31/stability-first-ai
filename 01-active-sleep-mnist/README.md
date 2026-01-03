# Active Sleep (Generative Replay) - MNIST

Basic experiment on MNIST demonstrating the concept of "active sleep" - generative replay of previous tasks through VAE.

## Description

The experiment demonstrates how catastrophic forgetting can be prevented by generating "dreams" from previous tasks using VAE, without the need to store real data.

## Architecture

- **SimpleMLP**: Simple MLP architecture for MNIST classification
- **VAE**: Variational autoencoder for generating "dreams"
- **Generative Replay**: Replaying generated data instead of real data

## Running

```bash
python active_sleep.py
```

## Results

Demonstrates the effectiveness of generative replay for preserving knowledge of previous tasks (~85-90% retention).

## Dependencies

- torch
- torchvision
- numpy
