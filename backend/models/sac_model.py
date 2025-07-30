from .rl_base import RLModelInterface

class SACModel(RLModelInterface):
    def generate_response(self, text: str) -> str:
        return f"A response to: {text}"

    def feedback(self, text: str, liked: bool) -> bool:
        score = 2 if liked else -1
        print(f"Model A received feedback {score} for: {text}")
        return True
