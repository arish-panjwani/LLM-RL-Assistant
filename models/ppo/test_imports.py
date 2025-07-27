#!/usr/bin/env python3
"""Test script to check imports"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from config.config import Config
    print("✓ Config import successful")
except Exception as e:
    print(f"✗ Config import failed: {e}")

try:
    from utils.groq_client import GroqClient
    print("✓ GroqClient import successful")
except Exception as e:
    print(f"✗ GroqClient import failed: {e}")

try:
    from environment.reward_calculator import RewardCalculator
    print("✓ RewardCalculator import successful")
except Exception as e:
    print(f"✗ RewardCalculator import failed: {e}")

try:
    from data.data_loader import DataLoader
    print("✓ DataLoader import successful")
except Exception as e:
    print(f"✗ DataLoader import failed: {e}")

print("Import test completed.") 