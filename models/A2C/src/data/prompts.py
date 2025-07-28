import random
from typing import List, Dict, Any, Optional

class PromptTemplates:
    """Collection of prompt templates for different use cases."""
    
    def __init__(self):
        """Initialize prompt templates."""
        self.templates = {
            'educational': self.get_educational_templates(),
            'problem_solving': self.get_problem_solving_templates(),
            'creative': self.get_creative_templates(),
            'professional': self.get_professional_templates(),
            'technical': self.get_technical_templates(),
            'health_wellness': self.get_health_wellness_templates(),
            'evaluation': self.get_evaluation_templates()
        }
    
    def get_educational_templates(self) -> List[str]:
        """Get educational prompt templates."""
        return [
            "What is {topic} and how does it work?",
            "Can you explain {topic} in simple terms?",
            "What are the key concepts in {topic}?",
            "How has {topic} evolved over time?",
            "What are the main applications of {topic}?",
            "What are the benefits and challenges of {topic}?",
            "How does {topic} compare to similar technologies?",
            "What is the future of {topic}?",
            "Who are the key figures in {topic}?",
            "What resources should I use to learn {topic}?",
            "What are the fundamental principles of {topic}?",
            "How can I get started with {topic}?",
            "What are the best practices in {topic}?",
            "What tools and technologies are used in {topic}?",
            "What are common misconceptions about {topic}?"
        ]
    
    def get_problem_solving_templates(self) -> List[str]:
        """Get problem-solving prompt templates."""
        return [
            "How do I approach {problem} effectively?",
            "What are the best practices for {problem}?",
            "What tools and techniques can help with {problem}?",
            "What are common mistakes to avoid in {problem}?",
            "How can I improve my {problem} skills?",
            "What resources should I use for {problem}?",
            "What is the step-by-step process for {problem}?",
            "How do I troubleshoot issues in {problem}?",
            "What metrics should I track for {problem}?",
            "How do I measure success in {problem}?",
            "What are the different approaches to {problem}?",
            "How do I prioritize tasks in {problem}?",
            "What are the key considerations for {problem}?",
            "How do I communicate about {problem}?",
            "What are the risks and mitigation strategies for {problem}?"
        ]
    
    def get_creative_templates(self) -> List[str]:
        """Get creative prompt templates."""
        return [
            "How can I improve my {task} skills?",
            "What techniques are useful for {task}?",
            "What inspires good {task}?",
            "How do I get started with {task}?",
            "What are the key elements of {task}?",
            "How can I make my {task} more engaging?",
            "What are the principles of effective {task}?",
            "How do I develop my own style in {task}?",
            "What tools and resources help with {task}?",
            "How do I overcome creative blocks in {task}?",
            "What are the different approaches to {task}?",
            "How do I get feedback on my {task}?",
            "What are the common challenges in {task}?",
            "How do I stay motivated in {task}?",
            "What are the trends in {task}?"
        ]
    
    def get_professional_templates(self) -> List[str]:
        """Get professional prompt templates."""
        return [
            "What are the key principles of {topic}?",
            "How can I improve my {topic} skills?",
            "What are common challenges in {topic}?",
            "What are the best practices for {topic}?",
            "How do I measure success in {topic}?",
            "What tools are essential for {topic}?",
            "How do I develop expertise in {topic}?",
            "What are the career opportunities in {topic}?",
            "How do I stay updated in {topic}?",
            "What are the ethical considerations in {topic}?",
            "How do I build credibility in {topic}?",
            "What are the industry standards for {topic}?",
            "How do I network effectively in {topic}?",
            "What are the certification requirements for {topic}?",
            "How do I balance work and life in {topic}?"
        ]
    
    def get_technical_templates(self) -> List[str]:
        """Get technical prompt templates."""
        return [
            "What are the fundamentals of {area}?",
            "How do I get started with {area}?",
            "What are the best practices for {area}?",
            "What tools should I learn for {area}?",
            "What are common pitfalls in {area}?",
            "How can I advance my {area} skills?",
            "What are the latest trends in {area}?",
            "How do I troubleshoot issues in {area}?",
            "What are the performance considerations in {area}?",
            "How do I scale solutions in {area}?",
            "What are the security implications of {area}?",
            "How do I integrate {area} with other technologies?",
            "What are the testing strategies for {area}?",
            "How do I deploy and maintain {area} solutions?",
            "What are the cost considerations for {area}?"
        ]
    
    def get_health_wellness_templates(self) -> List[str]:
        """Get health and wellness prompt templates."""
        return [
            "How can I improve my {topic}?",
            "What are the benefits of good {topic}?",
            "What are common mistakes in {topic}?",
            "How do I develop a {topic} routine?",
            "What resources can help with {topic}?",
            "How do I measure progress in {topic}?",
            "What are the scientific principles behind {topic}?",
            "How do I maintain consistency in {topic}?",
            "What are the different approaches to {topic}?",
            "How do I overcome obstacles in {topic}?",
            "What are the long-term effects of {topic}?",
            "How do I create a sustainable {topic} plan?",
            "What are the individual differences in {topic}?",
            "How do I adapt {topic} to my lifestyle?",
            "What are the professional resources for {topic}?"
        ]
    
    def get_evaluation_templates(self) -> List[str]:
        """Get evaluation and assessment prompt templates."""
        return [
            "Rate the quality of this {item} on a scale of 1-10",
            "Evaluate the effectiveness of this {item}",
            "Assess the strengths and weaknesses of this {item}",
            "What are the key metrics for evaluating this {item}?",
            "How does this {item} compare to alternatives?",
            "What improvements could be made to this {item}?",
            "What are the success criteria for this {item}?",
            "How do I measure the impact of this {item}?",
            "What are the benchmarks for this {item}?",
            "How do I validate the quality of this {item}?",
            "What are the risk factors for this {item}?",
            "How do I ensure consistency in this {item}?",
            "What are the compliance requirements for this {item}?",
            "How do I track progress in this {item}?",
            "What are the feedback mechanisms for this {item}?"
        ]
    
    def get_template(self, category: str, topic: str = None) -> str:
        """
        Get a random template from a category.
        
        Args:
            category: Template category
            topic: Topic to fill in the template
            
        Returns:
            Formatted template string
        """
        if category not in self.templates:
            raise ValueError(f"Unknown category: {category}")
        
        template = random.choice(self.templates[category])
        
        if topic:
            return template.format(topic=topic)
        else:
            return template
    
    def get_all_templates(self) -> Dict[str, List[str]]:
        """Get all templates."""
        return self.templates
    
    def add_template(self, category: str, template: str):
        """
        Add a new template to a category.
        
        Args:
            category: Template category
            template: Template string
        """
        if category not in self.templates:
            self.templates[category] = []
        
        self.templates[category].append(template)
    
    def get_categories(self) -> List[str]:
        """Get available template categories."""
        return list(self.templates.keys())
    
    def generate_prompt(self, 
                       category: str, 
                       topic: str = None,
                       style: str = 'default') -> str:
        """
        Generate a complete prompt.
        
        Args:
            category: Template category
            topic: Topic to fill in
            style: Prompt style ('default', 'detailed', 'simple')
            
        Returns:
            Generated prompt
        """
        base_prompt = self.get_template(category, topic)
        
        if style == 'detailed':
            return f"Please provide a comprehensive and detailed answer to: {base_prompt}"
        elif style == 'simple':
            return f"Please explain in simple terms: {base_prompt}"
        else:
            return base_prompt
    
    def get_contextual_prompt(self, 
                             base_prompt: str,
                             context: str = None,
                             constraints: List[str] = None,
                             format_requirements: str = None) -> str:
        """
        Create a contextual prompt with additional requirements.
        
        Args:
            base_prompt: Base prompt
            context: Additional context
            constraints: List of constraints
            format_requirements: Format requirements
            
        Returns:
            Contextual prompt
        """
        prompt_parts = [base_prompt]
        
        if context:
            prompt_parts.insert(0, f"Context: {context}")
        
        if constraints:
            constraints_text = ", ".join(constraints)
            prompt_parts.append(f"Constraints: {constraints_text}")
        
        if format_requirements:
            prompt_parts.append(f"Please format your response as: {format_requirements}")
        
        return " ".join(prompt_parts)
    
    def get_comparison_prompt(self, 
                             item1: str, 
                             item2: str, 
                             comparison_aspects: List[str] = None) -> str:
        """
        Create a comparison prompt.
        
        Args:
            item1: First item to compare
            item2: Second item to compare
            comparison_aspects: Aspects to compare
            
        Returns:
            Comparison prompt
        """
        base = f"Compare {item1} and {item2}"
        
        if comparison_aspects:
            aspects_text = ", ".join(comparison_aspects)
            base += f" in terms of: {aspects_text}"
        
        return base
    
    def get_analysis_prompt(self, 
                           subject: str, 
                           analysis_type: str = 'general') -> str:
        """
        Create an analysis prompt.
        
        Args:
            subject: Subject to analyze
            analysis_type: Type of analysis
            
        Returns:
            Analysis prompt
        """
        analysis_templates = {
            'general': f"Analyze {subject}",
            'detailed': f"Provide a detailed analysis of {subject}",
            'critical': f"Critically analyze {subject}",
            'comparative': f"Conduct a comparative analysis of {subject}",
            'trend': f"Analyze the trends in {subject}",
            'impact': f"Analyze the impact of {subject}",
            'risk': f"Conduct a risk analysis of {subject}"
        }
        
        return analysis_templates.get(analysis_type, analysis_templates['general']) 