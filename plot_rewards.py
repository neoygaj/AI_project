import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Adjust path to match your log directory
log_dir = "logs/dqn_breakout"
event_file = None

# Locate the latest event file
for root, _, files in os.walk(log_dir):
    for file in files:
        if file.startswith("events.out.tfevents"):
            event_file = os.path.join(root, file)
            break

if event_file is None:
    raise FileNotFoundError("No TensorBoard event file found in log_dir.")

# Load event data
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# Extract reward data
if "rollout/ep_rew_mean" not in ea.Tags()["scalars"]:
    raise KeyError("rollout/ep_rew_mean not found in the TensorBoard logs.")

rewards = ea.Scalars("rollout/ep_rew_mean")
steps = [event.step for event in rewards]
values = [event.value for event in rewards]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, values)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward (mean)")
plt.title("Training Progress: DQN on Breakout")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_rewards.png")
plt.show()
