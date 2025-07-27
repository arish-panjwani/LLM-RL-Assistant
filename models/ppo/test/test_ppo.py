import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

class TestRLPromptSystem(unittest.TestCase):
    """Comprehensive test suite for the RL prompt optimization system"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = Config()
        self.groq_client = GroqClient('')  # Mock client
        self.reward_calculator = RewardCalculator(self.groq_client, self.config)
        
        # Mock training data
        self.training_data = [
            {'query': 'Test query 1', 'context': '', 'type': 'test'},
            {'query': 'Test query 2', 'context': '', 'type': 'test'}
        ]
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        config = Config()
        self.assertIsInstance(config.EMBEDDING_DIM, int)
        self.assertIsInstance(config.CLARITY_WEIGHTS, dict)
        self.assertIn('lambda1', config.CLARITY_WEIGHTS)
    
    def test_groq_client_mock_response(self):
        """Test Groq client mock functionality"""
        client = GroqClient('')  # No API key, should use mock
        response = client.get_response("Test prompt")
        self.assertIsInstance(response, str)
        self.assertIn("Query:", response)
    
    def test_reward_calculator_clarity(self):
        """Test clarity reward calculation"""
        original = "What is AI?"
        modified = "Please explain artificial intelligence in detail"
        
        reward = self.reward_calculator.calculate_clarity_reward(original, modified)
        self.assertIsInstance(reward, float)
    
    def test_reward_calculator_relevance(self):
        """Test relevance reward calculation"""
        query = "What is machine learning?"
        response = "Machine learning is a subset of AI that enables computers to learn."
        
        reward = self.reward_calculator.calculate_relevance_reward(query, response)
        self.assertIsInstance(reward, float)
    
    def test_reward_calculator_hallucination(self):
        """Test hallucination penalty calculation"""
        response = "The capital of France is Paris."
        
        penalty = self.reward_calculator.calculate_hallucination_penalty(response)
        self.assertIsInstance(penalty, float)
        self.assertGreaterEqual(penalty, 0)
    
    def test_environment_creation(self):
        """Test environment creation and basic functionality"""
        env = PromptOptimizationEnv(
            self.groq_client, 
            self.reward_calculator, 
            self.training_data, 
            self.config
        )
        
        # Test reset
        obs, info = env.reset()
        self.assertEqual(obs.shape, (self.config.EMBEDDING_DIM,))
        
        # Test step
        action = np.random.uniform(-1, 1, self.config.ACTION_DIM)
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        self.assertIsInstance(reward, float)
        self.assertTrue(terminated)
        self.assertIn('original_prompt', info)
    
    def test_data_loader(self):
        """Test data loading functionality"""
        loader = DataLoader()
        persona_data, hh_data, truth_data = loader.load_training_data()
        
        self.assertIsInstance(persona_data, list)
        self.assertIsInstance(hh_data, list)
        self.assertIsInstance(truth_data, list)
        
        # Check data structure
        if persona_data:
            self.assertIn('query', persona_data[0])
            self.assertIn('type', persona_data[0])
    
    def test_deployment_server_initialization(self):
        """Test deployment server initialization"""
        server = DeploymentServer()
        
        self.assertIsNotNone(server.groq_client)
        self.assertIsNotNone(server.reward_calculator)
        self.assertIsInstance(server.models, dict)
    
    def test_prompt_optimization(self):
        """Test end-to-end prompt optimization"""
        server = DeploymentServer()
        
        test_prompt = "What is artificial intelligence?"
        result = server.optimize_prompt(test_prompt, 'PPO')
        
        self.assertIn('original_prompt', result)
        self.assertIn('optimized_prompt', result)
        self.assertIn('response', result)
        self.assertIn('metrics', result)
        
        # Check metrics structure
        metrics = result['metrics']
        self.assertIn('clarity_score', metrics)
        self.assertIn('relevance_score', metrics)
        self.assertIn('hallucination_penalty', metrics)
        self.assertIn('total_reward', metrics)

def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)

if __name__ == "__main__":
    run_tests()