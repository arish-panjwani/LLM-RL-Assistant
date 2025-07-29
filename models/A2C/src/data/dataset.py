import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class PromptDataset:
    """Dataset handler for prompt optimization training and evaluation."""
    
    def __init__(self, data_dir: str = "data/"):
        """
        Initialize the dataset handler.
        
        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Dataset splits
        self.train_prompts = []
        self.val_prompts = []
        self.test_prompts = []
        
        # Load or create datasets
        self.load_or_create_datasets()
        
        logger.info(f"Dataset initialized: {len(self.train_prompts)} train, "
                   f"{len(self.val_prompts)} val, {len(self.test_prompts)} test")
    
    def load_or_create_datasets(self):
        """Load existing datasets or create new ones."""
        train_path = os.path.join(self.processed_dir, "train_prompts.json")
        val_path = os.path.join(self.processed_dir, "val_prompts.json")
        test_path = os.path.join(self.processed_dir, "test_prompts.json")
        
        if all(os.path.exists(path) for path in [train_path, val_path, test_path]):
            self.load_datasets()
        else:
            self.create_datasets()
    
    def load_datasets(self):
        """Load existing datasets from files."""
        try:
            with open(os.path.join(self.processed_dir, "train_prompts.json"), 'r') as f:
                self.train_prompts = json.load(f)
            
            with open(os.path.join(self.processed_dir, "val_prompts.json"), 'r') as f:
                self.val_prompts = json.load(f)
            
            with open(os.path.join(self.processed_dir, "test_prompts.json"), 'r') as f:
                self.test_prompts = json.load(f)
                
            logger.info("Datasets loaded from files")
            
        except Exception as e:
            logger.warning(f"Failed to load datasets: {e}")
            self.create_datasets()
    
    def create_datasets(self):
        """Create new datasets."""
        # Generate comprehensive prompt dataset
        all_prompts = self.generate_comprehensive_prompts()
        
        # Split into train/val/test
        train_prompts, temp_prompts = train_test_split(
            all_prompts, test_size=0.3, random_state=42
        )
        val_prompts, test_prompts = train_test_split(
            temp_prompts, test_size=0.5, random_state=42
        )
        
        self.train_prompts = train_prompts
        self.val_prompts = val_prompts
        self.test_prompts = test_prompts
        
        # Save datasets
        self.save_datasets()
        
        logger.info("New datasets created and saved")
    
    def generate_comprehensive_prompts(self) -> List[str]:
        """Generate a comprehensive dataset of prompts."""
        prompts = []
        
        # Educational prompts
        educational_topics = [
            "machine learning", "artificial intelligence", "quantum computing",
            "blockchain", "cybersecurity", "data science", "robotics",
            "virtual reality", "augmented reality", "cloud computing",
            "internet of things", "5G technology", "renewable energy",
            "climate change", "genetics", "neuroscience", "psychology",
            "economics", "philosophy", "history", "geography", "astronomy",
            "chemistry", "physics", "biology", "mathematics", "statistics"
        ]
        
        for topic in educational_topics:
            prompts.extend([
                f"What is {topic}?",
                f"Explain {topic} in simple terms",
                f"How does {topic} work?",
                f"What are the applications of {topic}?",
                f"What are the benefits of {topic}?",
                f"What are the challenges in {topic}?",
                f"Compare {topic} with similar technologies",
                f"What is the future of {topic}?",
                f"Who are the key figures in {topic}?",
                f"What are the main concepts in {topic}?"
            ])
        
        # Problem-solving prompts
        problem_types = [
            "debugging code", "optimizing performance", "designing a system",
            "analyzing data", "making decisions", "solving equations",
            "writing algorithms", "creating models", "planning projects",
            "evaluating options", "troubleshooting issues", "improving processes"
        ]
        
        for problem in problem_types:
            prompts.extend([
                f"How do I approach {problem}?",
                f"What are the best practices for {problem}?",
                f"What tools can help with {problem}?",
                f"What are common mistakes in {problem}?",
                f"How can I improve my {problem} skills?",
                f"What resources should I use for {problem}?"
            ])
        
        # Creative prompts
        creative_tasks = [
            "writing a story", "creating artwork", "composing music",
            "designing a logo", "developing a character", "world-building",
            "brainstorming ideas", "solving puzzles", "inventing something",
            "planning an event", "organizing information", "presenting data"
        ]
        
        for task in creative_tasks:
            prompts.extend([
                f"How can I improve my {task}?",
                f"What techniques are useful for {task}?",
                f"What inspires good {task}?",
                f"How do I get started with {task}?",
                f"What are the key elements of {task}?",
                f"How can I make my {task} more engaging?"
            ])
        
        # Professional prompts
        professional_topics = [
            "leadership", "communication", "project management",
            "team building", "time management", "negotiation",
            "public speaking", "networking", "mentoring", "coaching",
            "strategic planning", "risk management", "quality assurance",
            "customer service", "sales", "marketing", "finance",
            "human resources", "operations", "research", "development"
        ]
        
        for topic in professional_topics:
            prompts.extend([
                f"What are the key principles of {topic}?",
                f"How can I improve my {topic} skills?",
                f"What are common challenges in {topic}?",
                f"What are the best practices for {topic}?",
                f"How do I measure success in {topic}?",
                f"What tools are essential for {topic}?"
            ])
        
        # Technical prompts
        technical_areas = [
            "programming", "database design", "system architecture",
            "network security", "software testing", "deployment",
            "monitoring", "scaling", "migration", "integration",
            "API design", "microservices", "containerization",
            "automation", "CI/CD", "DevOps", "agile", "scrum"
        ]
        
        for area in technical_areas:
            prompts.extend([
                f"What are the fundamentals of {area}?",
                f"How do I get started with {area}?",
                f"What are the best practices for {area}?",
                f"What tools should I learn for {area}?",
                f"What are common pitfalls in {area}?",
                f"How can I advance my {area} skills?"
            ])
        
        # Health and wellness prompts
        wellness_topics = [
            "exercise", "nutrition", "mental health", "sleep",
            "stress management", "meditation", "work-life balance",
            "relationships", "personal growth", "goal setting",
            "time management", "productivity", "motivation"
        ]
        
        for topic in wellness_topics:
            prompts.extend([
                f"How can I improve my {topic}?",
                f"What are the benefits of good {topic}?",
                f"What are common mistakes in {topic}?",
                f"How do I develop a {topic} routine?",
                f"What resources can help with {topic}?",
                f"How do I measure progress in {topic}?"
            ])
        
        # Remove duplicates and shuffle
        prompts = list(set(prompts))
        np.random.shuffle(prompts)
        
        return prompts
    
    def save_datasets(self):
        """Save datasets to files."""
        try:
            with open(os.path.join(self.processed_dir, "train_prompts.json"), 'w') as f:
                json.dump(self.train_prompts, f, indent=2)
            
            with open(os.path.join(self.processed_dir, "val_prompts.json"), 'w') as f:
                json.dump(self.val_prompts, f, indent=2)
            
            with open(os.path.join(self.processed_dir, "test_prompts.json"), 'w') as f:
                json.dump(self.test_prompts, f, indent=2)
                
            logger.info("Datasets saved to files")
            
        except Exception as e:
            logger.error(f"Failed to save datasets: {e}")
    
    def get_train_prompts(self) -> List[str]:
        """Get training prompts."""
        return self.train_prompts
    
    def get_val_prompts(self) -> List[str]:
        """Get validation prompts."""
        return self.val_prompts
    
    def get_test_prompts(self) -> List[str]:
        """Get test prompts."""
        return self.test_prompts
    
    def get_all_prompts(self) -> List[str]:
        """Get all prompts."""
        return self.train_prompts + self.val_prompts + self.test_prompts
    
    def add_prompts(self, prompts: List[str], split: str = 'train'):
        """
        Add new prompts to a specific split.
        
        Args:
            prompts: List of prompts to add
            split: Dataset split ('train', 'val', 'test')
        """
        if split == 'train':
            self.train_prompts.extend(prompts)
        elif split == 'val':
            self.val_prompts.extend(prompts)
        elif split == 'test':
            self.test_prompts.extend(prompts)
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Save updated datasets
        self.save_datasets()
        
        logger.info(f"Added {len(prompts)} prompts to {split} split")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        all_prompts = self.get_all_prompts()
        
        # Calculate prompt statistics
        prompt_lengths = [len(prompt.split()) for prompt in all_prompts]
        
        stats = {
            'total_prompts': len(all_prompts),
            'train_prompts': len(self.train_prompts),
            'val_prompts': len(self.val_prompts),
            'test_prompts': len(self.test_prompts),
            'avg_prompt_length': np.mean(prompt_lengths),
            'min_prompt_length': np.min(prompt_lengths),
            'max_prompt_length': np.max(prompt_lengths),
            'std_prompt_length': np.std(prompt_lengths)
        }
        
        return stats
    
    def export_to_csv(self, output_path: str = None):
        """Export dataset to CSV format."""
        if output_path is None:
            output_path = os.path.join(self.processed_dir, "prompt_dataset.csv")
        
        data = []
        for prompt in self.train_prompts:
            data.append({'prompt': prompt, 'split': 'train'})
        for prompt in self.val_prompts:
            data.append({'prompt': prompt, 'split': 'val'})
        for prompt in self.test_prompts:
            data.append({'prompt': prompt, 'split': 'test'})
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Dataset exported to {output_path}")
    
    def filter_prompts(self, 
                      min_length: int = 5,
                      max_length: int = 100,
                      keywords: List[str] = None,
                      exclude_keywords: List[str] = None) -> List[str]:
        """
        Filter prompts based on criteria.
        
        Args:
            min_length: Minimum word count
            max_length: Maximum word count
            keywords: Keywords that must be present
            exclude_keywords: Keywords to exclude
            
        Returns:
            Filtered list of prompts
        """
        filtered = []
        
        for prompt in self.get_all_prompts():
            word_count = len(prompt.split())
            
            # Check length
            if word_count < min_length or word_count > max_length:
                continue
            
            # Check keywords
            if keywords:
                if not any(keyword.lower() in prompt.lower() for keyword in keywords):
                    continue
            
            # Check exclude keywords
            if exclude_keywords:
                if any(keyword.lower() in prompt.lower() for keyword in exclude_keywords):
                    continue
            
            filtered.append(prompt)
        
        return filtered 