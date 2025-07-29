from typing import Dict, Callable, List
import random

class OptimizationStrategies:
    @staticmethod
    def get_strategies() -> Dict[int, List[Callable]]:
        return {
            # Analytical strategies
            0: [
                lambda p: f"Provide a detailed analysis of {p}, considering all key aspects",
                lambda p: f"What are the fundamental principles and mechanisms behind {p}?",
                lambda p: f"Examine {p} from multiple perspectives, including pros and cons"
            ],
            # Educational strategies
            1: [
                lambda p: f"Explain {p} as if teaching it to a beginner",
                lambda p: f"Break down {p} into simple, easy-to-understand components",
                lambda p: f"What are the most important concepts to understand about {p}?"
            ],
            # Comparative strategies
            2: [
                lambda p: f"Compare and contrast different aspects of {p}",
                lambda p: f"What are the similarities and differences in approaches to {p}?",
                lambda p: f"How does {p} relate to similar concepts in the field?"
            ],
            # Problem-solving strategies
            3: [
                lambda p: f"What are practical applications and real-world examples of {p}?",
                lambda p: f"How can {p} be implemented or used to solve problems?",
                lambda p: f"Describe a step-by-step approach to understanding {p}"
            ],
            # Critical thinking strategies
            4: [
                lambda p: f"What are the limitations and challenges associated with {p}?",
                lambda p: f"Evaluate the effectiveness and impact of {p}",
                lambda p: f"What are common misconceptions about {p}?"
            ]
        }

    @staticmethod
    def apply_optimization(prompt: str, action: int) -> str:
        strategies = OptimizationStrategies.get_strategies()
        strategy_list = strategies.get(action, strategies[0])
        chosen_strategy = random.choice(strategy_list)
        return chosen_strategy(prompt)