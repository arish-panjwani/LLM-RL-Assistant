import yaml
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for A2C prompt optimization."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables."""
        config = self.get_default_config()
        
        # Load from file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")
        
        # Override with environment variables
        config = self.override_with_env(config)
        
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            # Model configuration
            'model': {
                'state_dim': 50,
                'action_dim': 10,
                'hidden_dims': [128, 64, 32],
                'learning_rate': 0.001,
                'gamma': 0.99,
                'device': 'cpu'
            },
            
            # Training configuration
            'training': {
                'episodes': 1000,
                'max_steps_per_episode': 50,
                'batch_size': 32,
                'update_frequency': 10,
                'save_frequency': 100,
                'eval_frequency': 50,
                'early_stopping_patience': 50
            },
            
            # Environment configuration
            'environment': {
                'max_prompt_length': 500,
                'min_prompt_length': 10,
                'reward_weights': {
                    'coherence': 0.3,
                    'relevance': 0.3,
                    'clarity': 0.2,
                    'factual_accuracy': 0.2
                }
            },
            
            # Evaluation configuration
            'evaluation': {
                'metrics': ['cosine_similarity', 'sentiment', 'factual_accuracy'],
                'test_prompts_count': 100,
                'consistency_trials': 3
            },
            
            # API configuration
            'api': {
                'groq_api_key': os.getenv('GROQ_API_KEY', ''),
                'google_api_key': os.getenv('GOOGLE_API_KEY', ''),
                'google_cse_id': os.getenv('GOOGLE_CSE_ID', ''),
                'timeout': 30,
                'retry_attempts': 3
            },
            
            # Data configuration
            'data': {
                'dataset_path': 'data/processed/',
                'model_save_path': 'data/models/',
                'logs_path': 'logs/',
                'cache_dir': 'cache/'
            },
            
            # Logging configuration
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/a2c_training.log'
            }
        }
    
    def override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        env_mappings = {
            'GROQ_API_KEY': ('api', 'groq_api_key'),
            'GOOGLE_API_KEY': ('api', 'google_api_key'),
            'GOOGLE_CSE_ID': ('api', 'google_cse_id'),
            'A2C_LEARNING_RATE': ('model', 'learning_rate'),
            'A2C_EPISODES': ('training', 'episodes'),
            'A2C_DEVICE': ('model', 'device')
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                section, key = config_path
                if section in config and key in config[section]:
                    # Convert to appropriate type
                    original_type = type(config[section][key])
                    try:
                        if original_type == bool:
                            config[section][key] = env_value.lower() in ('true', '1', 'yes')
                        elif original_type == int:
                            config[section][key] = int(env_value)
                        elif original_type == float:
                            config[section][key] = float(env_value)
                        else:
                            config[section][key] = env_value
                    except ValueError:
                        print(f"Warning: Could not convert {env_var}={env_value} to {original_type}")
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = [
            'api.groq_api_key',
            'model.state_dim',
            'model.action_dim',
            'training.episodes'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"Error: Missing required configuration key: {key}")
                return False
        
        return True
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return self.config['training']
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        return self.config['environment']
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API-specific configuration."""
        return self.config['api']
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration."""
        return self.config['data']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging-specific configuration."""
        return self.config['logging'] 