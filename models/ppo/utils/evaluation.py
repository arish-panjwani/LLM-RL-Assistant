import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Dict, List, Any
import logging
from utils.groq_client import GroqClient
from environment.reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate and compare different RL models"""
    
    def __init__(self, models: Dict[str, Any], groq_client: GroqClient, 
                 reward_calculator: RewardCalculator, test_data: List[Dict]):
        self.models = models
        self.groq_client = groq_client
        self.reward_calculator = reward_calculator
        self.test_data = test_data
        
    def evaluate_all_models(self) -> pd.DataFrame:
        """Comprehensive evaluation of all models"""
        
        results = []
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            model_results = self._evaluate_single_model(model, model_name)
            results.extend(model_results)
        
        df = pd.DataFrame(results)
        
        # Generate evaluation report
        self._generate_evaluation_report(df)
        
        return df
    
    def _evaluate_single_model(self, model: Any, model_name: str) -> List[Dict]:
        """Evaluate a single model"""
        
        results = []
        
        for i, test_item in enumerate(self.test_data[:50]):  # Limit for testing
            try:
                # Get original query
                original_query = test_item['query']
                
                # Get model's embedding
                original_embedding = self.reward_calculator.embedding_model.encode([original_query])[0]
                
                # Get model prediction
                action, _ = model.predict(original_embedding, deterministic=True)
                
                # Apply action to get modified prompt
                modified_embedding = original_embedding + action * 0.1
                modified_prompt = self._embedding_to_prompt(modified_embedding, original_query)
                
                # Get responses
                original_response = self.groq_client.get_response(original_query)
                modified_response = self.groq_client.get_response(modified_prompt)
                
                # Calculate metrics
                clarity_score = self.reward_calculator.calculate_clarity_reward(original_query, modified_prompt)
                relevance_score = self.reward_calculator.calculate_relevance_reward(original_query, modified_response)
                hallucination_penalty = self.reward_calculator.calculate_hallucination_penalty(modified_response)
                total_reward = clarity_score + relevance_score - hallucination_penalty
                
                results.append({
                    'model': model_name,
                    'test_id': i,
                    'original_query': original_query,
                    'modified_prompt': modified_prompt,
                    'original_response': original_response,
                    'modified_response': modified_response,
                    'clarity_score': clarity_score,
                    'relevance_score': relevance_score,
                    'hallucination_penalty': hallucination_penalty,
                    'total_reward': total_reward,
                    'query_type': test_item.get('type', 'unknown')
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on test {i}: {e}")
                
        return results
    
    def _embedding_to_prompt(self, embedding: np.ndarray, original_query: str) -> str:
        """Convert embedding back to prompt (same as in environment)"""
        original_embedding = self.reward_calculator.embedding_model.encode([original_query])[0]
        
        similarity = np.dot(embedding, original_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(original_embedding)
        )
        
        if similarity > 0.95:
            return f"Please provide a clear and specific answer to: {original_query}"
        elif similarity > 0.8:
            return f"Can you explain in detail: {original_query}"
        else:
            return f"I need comprehensive information about: {original_query}"
    
    def _generate_evaluation_report(self, df: pd.DataFrame):
        """Generate comprehensive evaluation report with visualizations"""
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model comparison - Total reward
        model_rewards = df.groupby('model')['total_reward'].mean().sort_values(ascending=False)
        axes[0,0].bar(model_rewards.index, model_rewards.values)
        axes[0,0].set_title('Average Total Reward by Model')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Clarity scores comparison
        sns.boxplot(data=df, x='model', y='clarity_score', ax=axes[0,1])
        axes[0,1].set_title('Clarity Score Distribution by Model')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Relevance scores comparison
        sns.boxplot(data=df, x='model', y='relevance_score', ax=axes[0,2])
        axes[0,2].set_title('Relevance Score Distribution by Model')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Hallucination penalty comparison
        sns.boxplot(data=df, x='model', y='hallucination_penalty', ax=axes[1,0])
        axes[1,0].set_title('Hallucination Penalty by Model')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Performance by query type
        if 'query_type' in df.columns:
            pivot_data = df.pivot_table(values='total_reward', index='model', columns='query_type', aggfunc='mean')
            sns.heatmap(pivot_data, annot=True, cmap='viridis', ax=axes[1,1])
            axes[1,1].set_title('Performance by Query Type')
        else:
            axes[1,1].text(0.5, 0.5, 'No query type data', ha='center', va='center')
        
        # 6. Reward components correlation
        correlation_data = df[['clarity_score', 'relevance_score', 'hallucination_penalty', 'total_reward']].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('Reward Components Correlation')
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary statistics
        self._print_evaluation_summary(df)
    
    def _print_evaluation_summary(self, df: pd.DataFrame):
        """Print detailed evaluation summary"""
        
        print("\n" + "="*80)
        print("MULTI-MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # Overall performance ranking
        model_performance = df.groupby('model').agg({
            'total_reward': ['mean', 'std'],
            'clarity_score': 'mean',
            'relevance_score': 'mean',
            'hallucination_penalty': 'mean'
        }).round(4)
        
        print("\nModel Performance Ranking (by Total Reward):")
        print("-" * 50)
        
        for i, (model, row) in enumerate(model_performance.sort_values(('total_reward', 'mean'), ascending=False).iterrows(), 1):
            print(f"{i}. {model}:")
            print(f"   Total Reward: {row[('total_reward', 'mean')]:.4f} ± {row[('total_reward', 'std')]:.4f}")
            print(f"   Clarity: {row[('clarity_score', 'mean')]:.4f}")
            print(f"   Relevance: {row[('relevance_score', 'mean')]:.4f}")
            print(f"   Hallucination Penalty: {row[('hallucination_penalty', 'mean')]:.4f}")
            print()
        
        # Statistical significance tests could be added here
        print("Evaluation completed successfully!")
        print("Detailed results saved to 'model_evaluation_results.png'")
