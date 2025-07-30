# Saved Model Directory

This directory contains trained A2C model weights and checkpoints.

## Files

- `a2c_actor.pth` - Trained actor network weights (will be created after training)
- `README.md` - This file

## Usage

The model will automatically save trained weights to this directory during training. The main training script (`main.py`) will create the `a2c_actor.pth` file after successful training.

## Loading Models

Models can be loaded using:

```python
from model import A2CAgent

agent = A2CAgent(state_dim, action_dim, device)
agent.load("saved_model/a2c_actor.pth")
```

## Note

If no trained model exists, the system will start with an untrained model. Training is required to get meaningful prompt optimizations. 