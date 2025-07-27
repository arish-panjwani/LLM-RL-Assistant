import datasets
from transformers import AutoTokenizer
import pandas as pd
import nltk
import requests
import json
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
    
    def load_training_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load all training datasets"""
        logger.info("Loading training datasets...")
        
        # Load PersonaChat for dialogue training
        try:
            persona_data = datasets.load_dataset("bavard/personachat_truecased", split='train[:1000]')
            persona_samples = self._process_persona_data(persona_data)
        except Exception as e:
            logger.warning(f"Could not load PersonaChat: {e}")
            persona_samples = self._generate_mock_persona_data()
        
        # Load HH-RLHF for human feedback
        try:
            hh_data = datasets.load_dataset("Anthropic/hh-rlhf", split='train[:1000]')
            hh_samples = self._process_hh_data(hh_data)
        except Exception as e:
            logger.warning(f"Could not load HH-RLHF: {e}")
            hh_samples = self._generate_mock_hh_data()
        
        # Load TruthfulQA for hallucination detection
        try:
            truth_data = datasets.load_dataset("truthful_qa", "generation", split='validation[:500]')
            truth_samples = self._process_truth_data(truth_data)
        except Exception as e:
            logger.warning(f"Could not load TruthfulQA: {e}")
            truth_samples = self._generate_mock_truth_data()
        
        logger.info(f"Loaded {len(persona_samples)} persona samples, {len(hh_samples)} HH samples, {len(truth_samples)} truth samples")
        return persona_samples, hh_samples, truth_samples
    
    def _process_persona_data(self, data) -> List[Dict]:
        """Process PersonaChat data"""
        samples = []
        for item in data:
            if 'history' in item and len(item['history']) > 0:
                samples.append({
                    'query': item['history'][-1] if item['history'] else "Hello",
                    'context': " ".join(item['personality']) if 'personality' in item else "",
                    'type': 'conversation'
                })
        return samples[:500]  # Limit for training efficiency
    
    def _process_hh_data(self, data) -> List[Dict]:
        """Process HH-RLHF data"""
        samples = []
        for item in data:
            if 'chosen' in item:
                # Extract human message from chosen conversation
                messages = item['chosen'].split('\n\n')
                human_msgs = [msg for msg in messages if msg.startswith('Human:')]
                if human_msgs:
                    query = human_msgs[0].replace('Human:', '').strip()
                    samples.append({
                        'query': query,
                        'context': "",
                        'type': 'feedback'
                    })
        return samples[:500]
    
    def _process_truth_data(self, data) -> List[Dict]:
        """Process TruthfulQA data"""
        samples = []
        for item in data:
            if 'question' in item:
                samples.append({
                    'query': item['question'],
                    'context': "",
                    'correct_answers': item.get('correct_answers', []),
                    'type': 'factual'
                })
        return samples[:300]
    
    def _generate_mock_persona_data(self) -> List[Dict]:
        """Generate mock persona data if download fails"""
        return [
            {'query': 'Tell me about yourself', 'context': 'I am a helpful AI assistant', 'type': 'conversation'},
            {'query': 'What is the weather like?', 'context': 'I enjoy discussing weather', 'type': 'conversation'},
            {'query': 'How are you today?', 'context': 'I am always ready to help', 'type': 'conversation'},
            {'query': 'What can you help me with?', 'context': 'I can assist with various tasks', 'type': 'conversation'},
            {'query': 'Explain quantum physics', 'context': 'I love science topics', 'type': 'conversation'}
        ] * 20
    
    def _generate_mock_hh_data(self) -> List[Dict]:
        """Generate mock HH-RLHF data if download fails"""
        return [
            {'query': 'Explain machine learning in simple terms', 'context': '', 'type': 'feedback'},
            {'query': 'Write a Python function to sort a list', 'context': '', 'type': 'feedback'},
            {'query': 'What are the benefits of exercise?', 'context': '', 'type': 'feedback'},
            {'query': 'How does photosynthesis work?', 'context': '', 'type': 'feedback'},
            {'query': 'Recommend a good book to read', 'context': '', 'type': 'feedback'}
        ] * 20
    
    def _generate_mock_truth_data(self) -> List[Dict]:
        """Generate mock TruthfulQA data if download fails"""
        return [
            {'query': 'What is the capital of France?', 'context': '', 'correct_answers': ['Paris'], 'type': 'factual'},
            {'query': 'How many planets are in our solar system?', 'context': '', 'correct_answers': ['8', 'eight'], 'type': 'factual'},
            {'query': 'What year did World War II end?', 'context': '', 'correct_answers': ['1945'], 'type': 'factual'},
            {'query': 'What is the speed of light?', 'context': '', 'correct_answers': ['299,792,458 m/s'], 'type': 'factual'},
            {'query': 'Who wrote Romeo and Juliet?', 'context': '', 'correct_answers': ['William Shakespeare'], 'type': 'factual'}
        ] * 20