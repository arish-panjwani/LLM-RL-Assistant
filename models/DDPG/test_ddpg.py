
from model import DDPGAgent
import torch

state_dim = 384
action_dim = 384

agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=torch.device("cpu"))
dummy_state = torch.randn(state_dim)
action = agent.select_action(dummy_state.numpy())

print("Action Shape:", action.shape)
print("Action Preview:", action[:5])