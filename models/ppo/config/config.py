import os
from dataclasses import dataclass, field
from typing import Dict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

# Project root (one level up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
TENSORBOARD_PATH = os.path.join(PROJECT_ROOT, "tensorboard_logs")

@dataclass
class Config:
    # API Keys (read from environment variable)
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
    WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", '')

    # Model parameters
    EMBEDDING_DIM: int = 384
    ACTION_DIM: int = 384

    # Training parameters
    TOTAL_TIMESTEPS: int = 500
    N_ENVS: int = 2  # Reduced from 4 to 2
    EVAL_FREQ: int = 50
    
    # API Rate Limiting
    API_CALL_DELAY: float = 1.0  # Delay between API calls in seconds
    USE_CACHING: bool = True  # Cache API responses
    MAX_RETRIES: int = 3  # Maximum retries for failed API calls

    # Reward weights
    CLARITY_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'lambda1': 0.3,  # Similarity weight
        'lambda2': 0.2,  # Redundancy penalty weight
        'lambda3': 0.3,  # LLM rating weight
        'lambda4': 0.2   # Strategy diversity weight
    })
    RELEVANCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'alpha': 0.6,
        'beta': 0.4
    })
    HALLUCINATION_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'gamma': 0.5
    })

    # Paths (absolute)
    MODEL_SAVE_PATH: str = MODEL_SAVE_PATH
    LOG_PATH: str = LOG_PATH
    TENSORBOARD_PATH: str = TENSORBOARD_PATH

# Create a global config instance
config = Config()
