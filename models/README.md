# Reinforcement Learning Models

This folder contains the RL agents used to optimize prompts dynamically before sending them to the LLM.

## Models
- `ppo_model.py` – Proximal Policy Optimization
- `ddpg_model.py` – Deep Deterministic Policy Gradient
- `a2c_model.py` – Advantage Actor-Critic
- `sac_model.py` – Soft Actor-Critic
- `model_router.py` – Logic to route requests to the right model

## Features
- Trains on reward signals from human and AI feedback
- Learns how to rewrite prompts to maximize LLM response quality
- Supports both text and image-derived prompts

## Frameworks
- Stable-Baselines3
- PyTorch