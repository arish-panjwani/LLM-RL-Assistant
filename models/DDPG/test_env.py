from utils import load_pretrained_encoder, PromptEnvironment

encoder = load_pretrained_encoder()
env = PromptEnvironment(encoder)

state = env.reset()
print("Original Prompt Vector Shape:", state.shape)

# Simulate action (same as state for now)
action = state
next_state, reward, done, _ = env.step(action)

print("Refined Prompt Vector Shape:", next_state.shape)
print("Reward from Environment:", reward)
