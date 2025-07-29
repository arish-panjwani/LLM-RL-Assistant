import torch
from model import PPOAgent

def test_ppo():
    device = torch.device("cpu")
    state_dim = 384  # MiniLM embedding dimension
    action_dim = 384
    
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    # Test action selection
    state = torch.randn(1, state_dim)
    action, log_prob, value = agent.select_action(state)
    
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")
    print("✅ PPO model test passed!")

if __name__ == "__main__":
    test_ppo() 