from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment

def test_env():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    env = PromptEnvironment(encoder)
    
    test_prompt = "How do I cook rice?"
    state = env.encode(test_prompt)
    print(f"State shape: {state.shape}")
    print("✅ Environment test passed!")

if __name__ == "__main__":
    test_env() 