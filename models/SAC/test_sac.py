from model import SACAgent
import torch

state_dim = 384
action_dim = 384

device = torch.device("cpu")
agent = SACAgent(state_dim=state_dim, action_dim=action_dim, device=device)

dummy_state = torch.randn(state_dim)
action = agent.select_action(dummy_state.numpy())

print("✅ Action Shape:", action.shape)
print("✅ Action Sample:", action[:5])
