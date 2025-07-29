import numpy as np
import torch
import logging
from typing import Tuple, Dict, Any, Optional
import random

logger = logging.getLogger(__name__)

class PromptOptimizationEnv:
    """Reinforcement Learning Environment for Prompt Optimization - LIGHTWEIGHT VERSION with Real LLM Integration."""
    
    def __init__(self, groq_client=None, evaluator=None, config=None):
        self.groq_client = groq_client
        self.evaluator = evaluator
        self.config = config or {}
        
        # Environment parameters
        self.max_steps = config.get('max_steps', 10) if config else 10
        self.action_space = 5
        self.observation_space = 30
        
        # Training state
        self.current_step = 0
        self.current_prompt = ""
        self.original_prompt = ""
        self.episode_reward = 0.0
        
        # DIVERSE, DOMAIN-AGNOSTIC training prompts
        self.training_prompts = [
            # Short vague prompts - should learn add_clarity (Action 0)
            "ML", "AI", "neural", "training", "algorithm", "model", "data", "learning",
            "business", "health", "fitness", "cooking", "music", "sports", "travel",
            
            # Business/Finance prompts - should learn add_specificity (Action 1)
            "How to create a business plan?",
            "What are the best investment strategies?",
            "How to manage a project?",
            "What is market analysis?",
            "How to write a proposal?",
            "What are business requirements?",
            "How to conduct a meeting?",
            "What is financial planning?",
            
            # Health/Medical prompts - should learn add_context (Action 2)
            "What are the benefits of exercise?",
            "How to maintain a healthy diet?",
            "What vitamins should I take?",
            "How to improve sleep quality?",
            "What is stress management?",
            "How to start running?",
            "What foods boost energy?",
            "How to reduce anxiety?",
            
            # Lifestyle/Fashion prompts - should learn add_clarity (Action 0)
            "What are the latest fashion trends?",
            "How to organize a home office?",
            "What are good study habits?",
            "How to plan a party?",
            "What are time management tips?",
            "How to decorate a room?",
            "What are productivity hacks?",
            "How to maintain work-life balance?",
            
            # Sports/Entertainment prompts - should learn add_specificity (Action 1)
            "What are the rules of basketball?",
            "How to learn guitar for beginners?",
            "What are soccer strategies?",
            "How to play chess?",
            "What are tennis techniques?",
            "How to dance salsa?",
            "What are swimming strokes?",
            "How to paint with watercolors?",
            
            # Academic/Research prompts - should learn add_context (Action 2)
            "How to write a research paper?",
            "What is the scientific method?",
            "How to conduct an experiment?",
            "What is data analysis?",
            "How to give a presentation?",
            "What is critical thinking?",
            "How to solve math problems?",
            "What is literature review?",
            
            # Technical prompts - should learn add_specificity (Action 1)
            "What is machine learning?",
            "How does neural network work?",
            "Explain reinforcement learning",
            "What is deep learning?",
            "How to train a model?",
            "What is AI?",
            "Explain computer vision",
            "What is natural language processing?",
            
            # Long complex prompts - should learn simplify_language (Action 3)
            "What is the comprehensive methodology for implementing a deep reinforcement learning agent using actor-critic architecture with experience replay and prioritized sampling in a multi-agent environment?",
            "How do you analyze the performance characteristics of a convolutional neural network with residual connections and attention mechanisms for image classification tasks?",
            "Explain the theoretical foundations and practical implementation of transformer-based language models with self-attention mechanisms and positional encoding",
            "What are the advanced techniques for optimizing neural network architectures using automated machine learning and neural architecture search algorithms?",
            "How to develop a comprehensive business strategy that incorporates market analysis, competitive positioning, financial planning, and operational efficiency while considering external factors such as economic conditions and regulatory requirements?",
            "What is the complete methodology for conducting a thorough health assessment that includes physical examination, medical history review, diagnostic testing, risk factor analysis, and personalized treatment planning based on evidence-based medicine and patient preferences?"
        ]
        
        # Action mappings for prompt modification
        self.action_mappings = {
            0: "add_clarity",
            1: "add_specificity", 
            2: "add_context",
            3: "simplify_language",
            4: "no_change"
        }
        
        # Reward weights
        self.reward_weights = {
            'clarity': 0.3,
            'specificity': 0.3,
            'relevance': 0.2,
            'length': 0.1,
            'improvement': 0.1
        }
    
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode."""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Select a random prompt for this episode
        self.original_prompt = random.choice(self.training_prompts)
        self.current_prompt = self.original_prompt
        
        # Extract features for the initial state
        state = self.evaluator.extract_prompt_features(self.current_prompt).numpy()
        
        logger.info(f"Episode started with prompt: {self.original_prompt[:50]}...")
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.current_step += 1
        
        # Apply action to modify prompt
        modified_prompt = self.apply_action(self.current_prompt, action)
        
        # Get LLM response for evaluation (if available)
        llm_response = None
        if self.groq_client:
            try:
                llm_response = self.get_llm_response(modified_prompt)
                logger.debug(f"LLM Response: {llm_response[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to get LLM response: {e}")
        
        # Calculate reward using real evaluation
        reward, improvement = self.calculate_reward(
            self.original_prompt, 
            self.current_prompt, 
            modified_prompt, 
            action,
            llm_response
        )
        
        # Update current prompt
        self.current_prompt = modified_prompt
        self.episode_reward += reward
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Extract features for next state
        next_state = self.evaluator.extract_prompt_features(self.current_prompt).numpy()
        
        # Additional info
        info = {
            'action': action,
            'action_name': self.action_mappings.get(action, 'unknown'),
            'original_prompt': self.original_prompt,
            'current_prompt': self.current_prompt,
            'modified_prompt': modified_prompt,
            'reward': reward,
            'improvement': improvement,
            'llm_response': llm_response,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def apply_action(self, prompt: str, action: int) -> str:
        """Apply the selected action to modify the prompt."""
        action_name = self.action_mappings.get(action, "no_change")
        
        if action_name == "add_clarity":
            return f"Please provide a clear and detailed explanation of: {prompt}"
        elif action_name == "add_specificity":
            return f"Please provide specific and detailed information about: {prompt}"
        elif action_name == "add_context":
            return f"Please provide a comprehensive explanation with relevant context for: {prompt}"
        elif action_name == "simplify_language":
            return f"Please explain in simple terms that a beginner can understand: {prompt}"
        else:
            return prompt
    
    def get_llm_response(self, prompt: str) -> Optional[str]:
        """Get response from LLM for evaluation."""
        if not self.groq_client:
            return None
        
        try:
            # Use Groq API to get response
            response = self.groq_client.generate_response(
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            return response
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return None
    
    def calculate_reward(self, original_prompt: str, current_prompt: str, 
                        modified_prompt: str, action: int, llm_response: Optional[str] = None) -> Tuple[float, float]:
        """Calculate reward based on prompt improvement and LLM response quality."""
        
        # Base reward for taking action (encourage exploration of different actions)
        if action == 0:  # add_clarity
            base_reward = 0.15
        elif action == 1:  # add_specificity  
            base_reward = 0.15
        elif action == 2:  # add_context
            base_reward = 0.15
        elif action == 3:  # simplify_language
            base_reward = 0.1  # Slightly lower to encourage other actions
        else:  # no_change
            base_reward = 0.05
        
        # Evaluate prompt quality
        original_score = self.evaluator.evaluate_prompt(original_prompt)
        modified_score = self.evaluator.evaluate_prompt(modified_prompt)
        
        # Calculate improvement
        improvement = modified_score - original_score
        
        # LLM response evaluation (if available)
        llm_score = 0.0
        if llm_response and len(llm_response) > 10:
            try:
                # Evaluate response quality
                response_eval = self.evaluator.evaluate_response(llm_response, modified_prompt)
                llm_score = response_eval['overall_score']
                
                # Additional reward for good LLM responses
                if llm_score > 0.7:
                    llm_score *= 0.3  # Bonus for high-quality responses
            except Exception as e:
                logger.warning(f"Response evaluation failed: {e}")
        
        # Length penalty (prefer medium length)
        length_penalty = 0.0
        if len(modified_prompt) < 10:
            length_penalty = -0.1
        elif len(modified_prompt) > 300:
            length_penalty = -0.1
        
        # Relevance check (ensure prompt still relates to original)
        relevance_score = 0.0
        if self.evaluator:
            try:
                relevance_score = self.evaluator.cosine_similarity_score(original_prompt, modified_prompt)
            except:
                relevance_score = 0.5  # Default if calculation fails
        
        # Action-specific rewards based on prompt characteristics
        action_bonus = 0.0
        
        # Reward specific actions for specific prompt types
        if action == 0 and len(original_prompt.split()) < 5:  # add_clarity for short prompts
            action_bonus = 0.1
        elif action == 1 and 'technical' not in original_prompt.lower():  # add_specificity for non-technical
            action_bonus = 0.1
        elif action == 2 and 'ai' not in original_prompt.lower() and 'machine' not in original_prompt.lower():  # add_context for non-AI topics
            action_bonus = 0.1
        elif action == 3 and len(original_prompt.split()) > 8:  # simplify_language for long complex prompts
            action_bonus = 0.1
        elif action == 4 and len(original_prompt.split()) >= 5 and '?' in original_prompt:  # no_change for good prompts
            action_bonus = 0.1
        
        # Combine all components
        reward = (
            base_reward +
            improvement * self.reward_weights['improvement'] +
            llm_score * self.reward_weights['relevance'] +
            relevance_score * self.reward_weights['relevance'] +
            length_penalty +
            action_bonus
        )
        
        # Normalize reward
        reward = max(-1.0, min(1.0, reward))
        
        logger.debug(f"Reward calculation: base={base_reward:.3f}, improvement={improvement:.3f}, "
                    f"llm_score={llm_score:.3f}, relevance={relevance_score:.3f}, final={reward:.3f}")
        
        return reward, improvement
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            'current_step': self.current_step,
            'original_prompt': self.original_prompt,
            'current_prompt': self.current_prompt,
            'episode_reward': self.episode_reward,
            'max_steps': self.max_steps
        } 