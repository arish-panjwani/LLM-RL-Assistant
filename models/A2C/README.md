# A2C (Actor-Critic) Reinforcement Learning Model

**Kauthara's Assignment: Intelligent Conversational Assistant with RL-Optimized Prompt Engineering**

## Overview

A2C reinforcement learning model that dynamically optimizes prompts for better LLM responses. Learns from human feedback (70%) and automated metrics (30%).

## Quick Start

```bash
# Setup
conda activate a2c_env
pip install -r model_requirements.txt

# Test
python scripts/test_feedback.py
# Or web interface: cd demo && python app.py
```

## How It Works

1. **Input**: User enters a prompt
2. **A2C Optimization**: Model modifies prompt for better results
3. **LLM Response**: Groq API generates response
4. **Human Feedback**: Rate response (Good/Okay/Bad)
5. **Learning**: Model updates immediately from feedback

## Team Integration

### For Team Members - Simple Usage:

**Only need 2 files:**
- `data/models/a2c_domain_agnostic_best.pth` - Trained model
- `src/models/simple_interface.py` - Simple interface

```python
from src.models.simple_interface import SimpleA2CInterface

# Load model
interface = SimpleA2CInterface()

# See complete system (with feedback & LLM responses)
result = interface.optimize_prompt_with_feedback("How do I implement a neural network?")
print(f"Original: {result['original_prompt']}")
print(f"Optimized: {result['final_optimized_prompt']}")
print(f"LLM Response: {result['final_llm_response']}")

# Or just optimize prompts
result = interface.optimize_prompt("Your prompt here")
```

### Dependencies:
```bash
pip install torch numpy
```

## Core Files

### For Team Members:
- `data/models/a2c_domain_agnostic_best.pth` - Trained A2C model
- `src/models/simple_interface.py` - Simple interface
- `src/models/a2c_model.py` - Model architecture

### For Development:
- `src/models/prompt_optimizer.py` - Full implementation
- `scripts/` - Training and evaluation tools
- `demo/` - Web interface

## Status: Complete ✅

- A2C Model: Fully implemented and tested
- Real-Time Learning: Working with human feedback
- Team Integration: Ready for handoff
- Production Ready: Docker configured 