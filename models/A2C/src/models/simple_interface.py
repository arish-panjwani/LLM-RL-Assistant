"""
Simple Interface for Team Members
================================

This is a clean, simple interface that team members can use with just the trained .pth file.
Includes basic feedback handling and LLM responses to show how the A2C system works.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import json
import os

class SimpleA2CInterface:
    """Simple interface for team members to use the trained A2C model with feedback and LLM responses."""
    
    def __init__(self, model_path: str = "data/models/a2c_domain_agnostic_best.pth"):
        """
        Initialize with the trained model file.
        
        Args:
            model_path: Path to the trained .pth file
        """
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained A2C model."""
        try:
            from a2c_model import A2CModel
            
            # Basic model configuration
            model_config = {
                'state_dim': 30,
                'action_dim': 5,
                'hidden_dims': [64, 32],
                'learning_rate': 0.001,
                'gamma': 0.99
            }
            
            self.model = A2CModel(
                state_dim=model_config['state_dim'],
                action_dim=model_config['action_dim'],
                hidden_dims=model_config['hidden_dims'],
                learning_rate=model_config['learning_rate'],
                gamma=model_config['gamma'],
                device=self.device
            )
            
            # Load trained weights
            self.model.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def optimize_prompt_with_feedback(self, original_prompt: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Optimize a prompt using the trained A2C model with feedback and LLM responses.
        
        Args:
            original_prompt: The original prompt to optimize
            max_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimization results including LLM responses and feedback
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        current_prompt = original_prompt
        optimization_history = []
        
        print(f"\n🚀 Starting A2C Optimization for: '{original_prompt}'")
        print("=" * 60)
        
        for iteration in range(max_iterations):
            print(f"\n📝 Iteration {iteration + 1}/{max_iterations}")
            print(f"Current prompt: {current_prompt[:80]}{'...' if len(current_prompt) > 80 else ''}")
            
            # Extract features from current prompt
            features = self._extract_features(current_prompt)
            
            # Get model's action
            with torch.no_grad():
                action_probs = self.model.actor(features)
                action = torch.argmax(action_probs).item()
            
            # Apply the action to modify the prompt
            modified_prompt = self._apply_action(current_prompt, action)
            action_name = self._get_action_name(action)
            
            print(f"🤖 A2C Action: {action_name}")
            print(f"Optimized prompt: {modified_prompt[:80]}{'...' if len(modified_prompt) > 80 else ''}")
            
            # Get LLM response (simulated or real)
            llm_response = self._get_llm_response(modified_prompt)
            print(f"🤖 LLM Response: {llm_response[:60]}...")
            
            # Get automated evaluation score
            evaluation_score = self._evaluate_response_quality(modified_prompt, llm_response)
            print(f"📊 Automated Score: {evaluation_score:.3f}")
            
            # Simulate human feedback (team members can replace this with real feedback)
            human_feedback = self._simulate_human_feedback(modified_prompt, llm_response)
            print(f"👤 Human Feedback: {human_feedback:.1f}")
            
            # Calculate combined reward
            reward = 0.7 * human_feedback + 0.3 * evaluation_score
            print(f"🎯 Combined Reward: {reward:.3f}")
            print("-" * 40)
            
            optimization_history.append({
                'iteration': iteration + 1,
                'action': action,
                'action_name': action_name,
                'original_prompt': current_prompt,
                'optimized_prompt': modified_prompt,
                'llm_response': llm_response,
                'evaluation_score': evaluation_score,
                'human_feedback': human_feedback,
                'reward': reward
            })
            
            current_prompt = modified_prompt
        
        print("\n" + "=" * 60)
        print("✅ Optimization Complete!")
        
        return {
            'original_prompt': original_prompt,
            'final_optimized_prompt': current_prompt,
            'optimization_history': optimization_history,
            'total_iterations': max_iterations,
            'final_llm_response': optimization_history[-1]['llm_response'],
            'final_reward': optimization_history[-1]['reward']
        }
    
    def optimize_prompt(self, original_prompt: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Simple prompt optimization without feedback (for basic usage).
        
        Args:
            original_prompt: The original prompt to optimize
            max_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        current_prompt = original_prompt
        optimization_history = []
        
        for iteration in range(max_iterations):
            # Extract features from current prompt
            features = self._extract_features(current_prompt)
            
            # Get model's action
            with torch.no_grad():
                action_probs = self.model.actor(features)
                action = torch.argmax(action_probs).item()
            
            # Apply the action to modify the prompt
            modified_prompt = self._apply_action(current_prompt, action)
            
            optimization_history.append({
                'iteration': iteration + 1,
                'action': action,
                'action_name': self._get_action_name(action),
                'original_prompt': current_prompt,
                'optimized_prompt': modified_prompt
            })
            
            current_prompt = modified_prompt
        
        return {
            'original_prompt': original_prompt,
            'final_optimized_prompt': current_prompt,
            'optimization_history': optimization_history,
            'total_iterations': max_iterations
        }
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get LLM response (simulated for demo, can be replaced with real API)."""
        # Simulated LLM responses for demonstration
        responses = {
            "neural network": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information. To implement one, you typically need to define the architecture, choose activation functions, and train with data using backpropagation.",
            "machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions.",
            "deep learning": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition.",
            "reinforcement learning": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward. It's commonly used in game playing, robotics, and autonomous systems."
        }
        
        # Check for keywords and return appropriate response
        prompt_lower = prompt.lower()
        for keyword, response in responses.items():
            if keyword in prompt_lower:
                return response
        
        # Default response
        return f"This is a simulated response to: '{prompt}'. In a real implementation, this would be replaced with an actual LLM API call (like Groq, OpenAI, etc.)."
    
    def _evaluate_response_quality(self, prompt: str, response: str) -> float:
        """Evaluate response quality using simple metrics."""
        # Simple evaluation based on response length and relevance
        if not response or not prompt:
            return 0.0
        
        # Length score (prefer longer, detailed responses)
        length_score = min(len(response) / 200.0, 1.0)
        
        # Relevance score (check for common words)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        common_words = prompt_words.intersection(response_words)
        relevance_score = len(common_words) / max(len(prompt_words), 1)
        
        # Combined score
        return (length_score + relevance_score) / 2.0
    
    def _simulate_human_feedback(self, prompt: str, response: str) -> float:
        """Simulate human feedback (team members can replace with real feedback)."""
        # Simple simulation based on response quality
        quality = self._evaluate_response_quality(prompt, response)
        
        # Map quality to human feedback (0.0 = Bad, 0.5 = Okay, 1.0 = Good)
        if quality > 0.7:
            return 1.0  # Good
        elif quality > 0.4:
            return 0.5  # Okay
        else:
            return 0.0  # Bad
    
    def _extract_features(self, prompt: str) -> torch.Tensor:
        """Extract features from prompt text."""
        # Simple feature extraction (same as in your original model)
        features = []
        
        # Length features
        features.append(len(prompt) / 1000.0)  # Normalized length
        features.append(len(prompt.split()) / 100.0)  # Word count
        
        # Character features
        features.append(prompt.count('?') / 10.0)
        features.append(prompt.count('!') / 10.0)
        features.append(prompt.count('.') / 10.0)
        
        # Word features
        words = prompt.lower().split()
        features.append(len([w for w in words if len(w) > 6]) / len(words) if words else 0)
        features.append(len([w for w in words if w.endswith('ing')]) / len(words) if words else 0)
        features.append(len([w for w in words if w.endswith('ed')]) / len(words) if words else 0)
        
        # Question features
        features.append(1.0 if '?' in prompt else 0.0)
        features.append(1.0 if any(w in prompt.lower() for w in ['what', 'how', 'why', 'when', 'where', 'who']) else 0.0)
        
        # Politeness features
        polite_words = ['please', 'thank', 'sorry', 'excuse', 'would you', 'could you']
        features.append(1.0 if any(word in prompt.lower() for word in polite_words) else 0.0)
        
        # Technical features
        tech_words = ['api', 'function', 'code', 'algorithm', 'data', 'system']
        features.append(1.0 if any(word in prompt.lower() for word in tech_words) else 0.0)
        
        # Fill remaining features with zeros
        while len(features) < 30:
            features.append(0.0)
        
        return torch.tensor(features[:30], dtype=torch.float32, device=self.device)
    
    def _apply_action(self, prompt: str, action: int) -> str:
        """Apply the model's action to modify the prompt."""
        actions = {
            0: lambda p: p + " Please provide a detailed explanation.",
            1: lambda p: p + " Can you give me a step-by-step guide?",
            2: lambda p: "I need help with: " + p,
            3: lambda p: p.replace("?", "? I would appreciate a comprehensive answer.") if "?" in p else p + "?",
            4: lambda p: p + " Please include examples and best practices."
        }
        
        if action in actions:
            return actions[action](prompt)
        return prompt
    
    def _get_action_name(self, action: int) -> str:
        """Get the name of the action."""
        action_names = {
            0: "Add Detail Request",
            1: "Add Step-by-Step Request", 
            2: "Add Help Context",
            3: "Add Comprehensive Request",
            4: "Add Examples Request"
        }
        return action_names.get(action, f"Action {action}")
    
    def batch_optimize(self, prompts: list, max_iterations: int = 3) -> list:
        """
        Optimize multiple prompts.
        
        Args:
            prompts: List of prompts to optimize
            max_iterations: Number of optimization iterations per prompt
            
        Returns:
            List of optimization results
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Optimizing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            result = self.optimize_prompt(prompt, max_iterations)
            results.append(result)
        return results


# Simple usage example
if __name__ == "__main__":
    # Example usage for team members
    interface = SimpleA2CInterface()
    
    # Test with full feedback system
    print("🎯 Testing A2C System with Feedback and LLM Responses")
    print("=" * 60)
    
    test_prompt = "How do I implement a neural network?"
    result = interface.optimize_prompt_with_feedback(test_prompt)
    
    print(f"\n📊 Final Results:")
    print(f"Original: {result['original_prompt']}")
    print(f"Optimized: {result['final_optimized_prompt']}")
    print(f"Final LLM Response: {result['final_llm_response'][:100]}...")
    print(f"Final Reward: {result['final_reward']:.3f}")
    
    print(f"\n🔄 Optimization History:")
    for step in result['optimization_history']:
        print(f"Iteration {step['iteration']}: {step['action_name']}")
        print(f"  Reward: {step['reward']:.3f}") 