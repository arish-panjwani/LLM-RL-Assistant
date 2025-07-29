#!/usr/bin/env python3
"""
Metrics Analysis Script for Thejaswi (Data Analyst)
This script analyzes the optimization history and provides metrics insights.
"""

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.evaluation_metrics import PromptEvaluator

class MetricsAnalyzer:
    """Analyzes metrics data from optimization history."""
    
    def __init__(self, history_file: str = "demo/optimization_history.json"):
        self.history_file = Path(history_file)
        self.evaluator = PromptEvaluator()
        self.data = []
        
    def load_history(self) -> List[Dict[str, Any]]:
        """Load optimization history from file."""
        if not self.history_file.exists():
            print(f"⚠️ History file not found: {self.history_file}")
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            print(f"✅ Loaded {len(history)} optimization records")
            return history
        except Exception as e:
            print(f"❌ Error loading history: {e}")
            return []
    
    def calculate_metrics_for_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate detailed metrics for each optimization record."""
        enhanced_data = []
        
        for record in history:
            original_prompt = record.get('original_prompt', '')
            optimized_prompt = record.get('optimized_prompt', '')
            
            # Get LLM responses (if available)
            original_response = record.get('original_response', '')
            optimized_response = record.get('optimized_response', '')
            
            # Calculate metrics for original response
            if original_response and original_response != "LLM response will be available when API is configured":
                original_metrics = self.evaluator.evaluate_response(original_response, original_prompt)
            else:
                original_metrics = {
                    'overall_score': record.get('initial_score', 0.5),
                    'sentiment_score': 0.0,
                    'hallucination_score': 0.5,
                    'diversity_score': 0.5,
                    'cosine_similarity': 0.5,
                    'length_score': 0.5
                }
            
            # Calculate metrics for optimized response
            if optimized_response and optimized_response != "LLM response will be available when API is configured":
                optimized_metrics = self.evaluator.evaluate_response(optimized_response, optimized_prompt)
            else:
                optimized_metrics = {
                    'overall_score': record.get('final_score', 0.5),
                    'sentiment_score': 0.0,
                    'hallucination_score': 0.5,
                    'diversity_score': 0.5,
                    'cosine_similarity': 0.5,
                    'length_score': 0.5
                }
            
            # Create enhanced record
            enhanced_record = {
                'id': record.get('id', 0),
                'timestamp': record.get('timestamp', ''),
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'action_name': record.get('action_name', 'Unknown'),
                'improvement': record.get('improvement', 0),
                
                # Original metrics
                'original_overall': original_metrics.get('overall_score', 0),
                'original_sentiment': original_metrics.get('sentiment_score', 0),
                'original_factual': 1 - original_metrics.get('hallucination_score', 0.5),
                'original_diversity': original_metrics.get('diversity_score', 0.5),
                'original_cosine': original_metrics.get('cosine_similarity', 0.5),
                'original_length': original_metrics.get('length_score', 0.5),
                
                # Optimized metrics
                'optimized_overall': optimized_metrics.get('overall_score', 0),
                'optimized_sentiment': optimized_metrics.get('sentiment_score', 0),
                'optimized_factual': 1 - optimized_metrics.get('hallucination_score', 0.5),
                'optimized_diversity': optimized_metrics.get('diversity_score', 0.5),
                'optimized_cosine': optimized_metrics.get('cosine_similarity', 0.5),
                'optimized_length': optimized_metrics.get('length_score', 0.5),
                
                # Improvements
                'overall_improvement': optimized_metrics.get('overall_score', 0) - original_metrics.get('overall_score', 0),
                'sentiment_improvement': optimized_metrics.get('sentiment_score', 0) - original_metrics.get('sentiment_score', 0),
                'factual_improvement': (1 - optimized_metrics.get('hallucination_score', 0.5)) - (1 - original_metrics.get('hallucination_score', 0.5)),
                'diversity_improvement': optimized_metrics.get('diversity_score', 0.5) - original_metrics.get('diversity_score', 0.5),
                'cosine_improvement': optimized_metrics.get('cosine_similarity', 0.5) - original_metrics.get('cosine_similarity', 0.5),
                'length_improvement': optimized_metrics.get('length_score', 0.5) - original_metrics.get('length_score', 0.5),
            }
            
            enhanced_data.append(enhanced_record)
        
        return enhanced_data
    
    def generate_metrics_report(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        if not data:
            return {"error": "No data available"}
        
        df = pd.DataFrame(data)
        
        report = {
            'summary': {
                'total_optimizations': len(data),
                'average_overall_improvement': df['overall_improvement'].mean(),
                'average_cosine_improvement': df['cosine_improvement'].mean(),
                'average_sentiment_improvement': df['sentiment_improvement'].mean(),
                'average_factual_improvement': df['factual_improvement'].mean(),
                'average_diversity_improvement': df['diversity_improvement'].mean(),
            },
            
            'metrics_analysis': {
                'cosine_similarity': {
                    'original_avg': df['original_cosine'].mean(),
                    'optimized_avg': df['optimized_cosine'].mean(),
                    'improvement_avg': df['cosine_improvement'].mean(),
                    'best_improvement': df['cosine_improvement'].max(),
                    'worst_improvement': df['cosine_improvement'].min(),
                },
                'sentiment_score': {
                    'original_avg': df['original_sentiment'].mean(),
                    'optimized_avg': df['optimized_sentiment'].mean(),
                    'improvement_avg': df['sentiment_improvement'].mean(),
                    'best_improvement': df['sentiment_improvement'].max(),
                    'worst_improvement': df['sentiment_improvement'].min(),
                },
                'factual_accuracy': {
                    'original_avg': df['original_factual'].mean(),
                    'optimized_avg': df['optimized_factual'].mean(),
                    'improvement_avg': df['factual_improvement'].mean(),
                    'best_improvement': df['factual_improvement'].max(),
                    'worst_improvement': df['factual_improvement'].min(),
                },
                'lexical_diversity': {
                    'original_avg': df['original_diversity'].mean(),
                    'optimized_avg': df['optimized_diversity'].mean(),
                    'improvement_avg': df['diversity_improvement'].mean(),
                    'best_improvement': df['diversity_improvement'].max(),
                    'worst_improvement': df['diversity_improvement'].min(),
                }
            },
            
            'action_analysis': df.groupby('action_name').agg({
                'overall_improvement': ['mean', 'count'],
                'cosine_improvement': 'mean',
                'sentiment_improvement': 'mean',
                'factual_improvement': 'mean',
                'diversity_improvement': 'mean'
            }).round(3).to_dict(),
            
            'top_improvements': df.nlargest(5, 'overall_improvement')[['id', 'original_prompt', 'overall_improvement', 'cosine_improvement', 'action_name']].to_dict('records'),
            
            'data_for_visualization': {
                'metrics': ['Cosine Similarity', 'Sentiment', 'Factual Accuracy', 'Diversity'],
                'original_scores': [
                    df['original_cosine'].mean(),
                    df['original_sentiment'].mean(),
                    df['original_factual'].mean(),
                    df['original_diversity'].mean()
                ],
                'optimized_scores': [
                    df['optimized_cosine'].mean(),
                    df['optimized_sentiment'].mean(),
                    df['optimized_factual'].mean(),
                    df['optimized_diversity'].mean()
                ]
            }
        }
        
        return report
    
    def save_metrics_report(self, report: Dict[str, Any], output_file: str = "data/metrics_analysis_report.json"):
        """Save metrics report to file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✅ Metrics report saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the metrics analysis."""
        print("\n" + "="*60)
        print("📊 METRICS ANALYSIS SUMMARY")
        print("="*60)
        
        if 'error' in report:
            print(f"❌ {report['error']}")
            return
        
        summary = report['summary']
        print(f"📈 Total Optimizations: {summary['total_optimizations']}")
        print(f"📊 Average Overall Improvement: {summary['average_overall_improvement']:.3f}")
        print(f"🔗 Average Cosine Similarity Improvement: {summary['average_cosine_improvement']:.3f}")
        print(f"😊 Average Sentiment Improvement: {summary['average_sentiment_improvement']:.3f}")
        print(f"✅ Average Factual Accuracy Improvement: {summary['average_factual_improvement']:.3f}")
        print(f"📝 Average Diversity Improvement: {summary['average_diversity_improvement']:.3f}")
        
        print("\n🏆 TOP 3 IMPROVEMENTS:")
        for i, improvement in enumerate(report['top_improvements'][:3], 1):
            print(f"{i}. ID {improvement['id']}: {improvement['overall_improvement']:.3f} improvement")
            print(f"   Prompt: {improvement['original_prompt'][:50]}...")
            print(f"   Action: {improvement['action_name']}")
        
        print("\n🎯 METRICS BREAKDOWN:")
        metrics = report['metrics_analysis']
        for metric_name, data in metrics.items():
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print(f"  Original Avg: {data['original_avg']:.3f}")
            print(f"  Optimized Avg: {data['optimized_avg']:.3f}")
            print(f"  Improvement: {data['improvement_avg']:.3f}")

def main():
    """Main function to run metrics analysis."""
    print("🔍 A2C Metrics Analysis for Thejaswi")
    print("="*50)
    
    analyzer = MetricsAnalyzer()
    
    # Load history
    history = analyzer.load_history()
    if not history:
        print("❌ No data to analyze")
        return
    
    # Calculate metrics
    print("📊 Calculating detailed metrics...")
    enhanced_data = analyzer.calculate_metrics_for_history(history)
    
    # Generate report
    print("📈 Generating metrics report...")
    report = analyzer.generate_metrics_report(enhanced_data)
    
    # Save report
    analyzer.save_metrics_report(report)
    
    # Print summary
    analyzer.print_summary(report)
    
    print("\n✅ Analysis complete! Thejaswi can now use this data for her metrics design task.")

if __name__ == "__main__":
    main() 