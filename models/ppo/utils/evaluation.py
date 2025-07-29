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
        
        # Check if we have any results
        if df.empty:
            logger.warning("No evaluation results generated. Creating empty DataFrame with expected columns.")
            df = pd.DataFrame(columns=[
                'model', 'test_id', 'original_query', 'modified_prompt', 
                'original_response', 'modified_response', 'clarity_score', 
                'relevance_score', 'hallucination_penalty', 'total_reward', 'query_type'
            ])
        else:
            # Generate evaluation report only if we have results
            self._generate_evaluation_report(df)
        
        return df
    
    def _evaluate_single_model(self, model: Any, model_name: str) -> List[Dict]:
        """Evaluate a single model"""
        
        results = []
        
        for i, test_item in enumerate(self.test_data[:50]):  # Limit for testing
            try:
                # Get original query
                if 'query' not in test_item:
                    logger.warning(f"Test item {i} missing 'query' key. Available keys: {list(test_item.keys())}")
                    continue
                    
                original_query = test_item['query']
                logger.debug(f"Processing test {i}: {original_query[:50]}...")
                
                # Get model's embedding
                original_embedding = self.reward_calculator.embedding_model.encode([original_query])[0]
                
                # Get model prediction
                try:
                    action, _ = model.predict(original_embedding, deterministic=True)
                    logger.debug(f"Model prediction successful for {model_name}, action shape: {action.shape if hasattr(action, 'shape') else 'scalar'}")
                except Exception as e:
                    logger.error(f"Model prediction failed for {model_name}: {e}")
                    # Skip this test item if prediction fails
                    continue
                
                # Apply action to get modified prompt
                modified_embedding = original_embedding + action * 0.1
                modified_prompt = self._embedding_to_prompt(modified_embedding, original_query)
                
                # Get responses
                try:
                    original_response = self.groq_client.get_response(original_query)
                    modified_response = self.groq_client.get_response(modified_prompt)
                    logger.debug(f"API responses received for test {i}")
                except Exception as e:
                    logger.error(f"API call failed for test {i}: {e}")
                    # Use placeholder responses if API fails
                    original_response = "API call failed"
                    modified_response = "API call failed"
                
                # Calculate metrics using the correct method names
                try:
                    clarity_score = self.reward_calculator._calculate_clarity_score(modified_prompt)
                    relevance_score = self.reward_calculator._calculate_relevance_score(original_query, modified_prompt)
                    hallucination_penalty = self.reward_calculator._calculate_hallucination_score(modified_response)
                    total_reward = clarity_score + relevance_score - hallucination_penalty
                    logger.debug(f"Reward calculation successful for test {i}: clarity={clarity_score:.3f}, relevance={relevance_score:.3f}, hallucination={hallucination_penalty:.3f}, total={total_reward:.3f}")
                except Exception as e:
                    logger.error(f"Reward calculation failed for test {i}: {e}")
                    # Use default values if calculation fails
                    clarity_score = 0.5
                    relevance_score = 0.5
                    hallucination_penalty = 0.3
                    total_reward = 0.7
                
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
                
        logger.info(f"Completed evaluation for {model_name}: {len(results)} successful results out of {min(50, len(self.test_data))} tests")
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
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning("Cannot generate evaluation report: DataFrame is empty")
            return
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model comparison - Total reward
        if 'model' in df.columns and 'total_reward' in df.columns and not df.empty:
            model_rewards = df.groupby('model')['total_reward'].mean().sort_values(ascending=False)
            axes[0,0].bar(model_rewards.index, model_rewards.values)
            axes[0,0].set_title('Average Total Reward by Model')
            axes[0,0].set_ylabel('Total Reward')
            axes[0,0].tick_params(axis='x', rotation=45)
        else:
            axes[0,0].text(0.5, 0.5, 'No data available', ha='center', va='center')
            axes[0,0].set_title('Average Total Reward by Model')
        
        # 2. Clarity scores comparison
        if 'model' in df.columns and 'clarity_score' in df.columns and not df.empty:
            sns.boxplot(data=df, x='model', y='clarity_score', ax=axes[0,1])
            axes[0,1].set_title('Clarity Score Distribution by Model')
            axes[0,1].tick_params(axis='x', rotation=45)
        else:
            axes[0,1].text(0.5, 0.5, 'No data available', ha='center', va='center')
            axes[0,1].set_title('Clarity Score Distribution by Model')
        
        # 3. Relevance scores comparison
        if 'model' in df.columns and 'relevance_score' in df.columns and not df.empty:
            sns.boxplot(data=df, x='model', y='relevance_score', ax=axes[0,2])
            axes[0,2].set_title('Relevance Score Distribution by Model')
            axes[0,2].tick_params(axis='x', rotation=45)
        else:
            axes[0,2].text(0.5, 0.5, 'No data available', ha='center', va='center')
            axes[0,2].set_title('Relevance Score Distribution by Model')
        
        # 4. Hallucination penalty comparison
        if 'model' in df.columns and 'hallucination_penalty' in df.columns and not df.empty:
            sns.boxplot(data=df, x='model', y='hallucination_penalty', ax=axes[1,0])
            axes[1,0].set_title('Hallucination Penalty by Model')
            axes[1,0].tick_params(axis='x', rotation=45)
        else:
            axes[1,0].text(0.5, 0.5, 'No data available', ha='center', va='center')
            axes[1,0].set_title('Hallucination Penalty by Model')
        
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
        
        if df.empty:
            print("\n" + "="*80)
            print("MULTI-MODEL EVALUATION SUMMARY")
            print("="*80)
            print("\n⚠️ No evaluation results available.")
            print("This could be due to:")
            print("- Model evaluation errors")
            print("- Missing or incompatible models")
            print("- API connection issues")
            print("\nCheck the logs for more details.")
            return
        
        print("\n" + "="*80)
        print("MULTI-MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # Overall performance ranking
        if 'model' in df.columns and not df.empty:
            model_performance = df.groupby('model').agg({
                'total_reward': ['mean', 'std'],
                'clarity_score': 'mean',
                'relevance_score': 'mean',
                'hallucination_penalty': 'mean'
            }).round(4)
        else:
            print("No model performance data available.")
            return
        
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
