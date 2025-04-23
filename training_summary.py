import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# log_dir = "logs/dqn_breakout/{run_name}"
log_dir = "logs/dqn_breakout"
output_pdf = "training_summary.pdf"

tags = {
    "rollout/ep_rew_mean": "Mean Episode Reward",
    "rollout/ep_len_mean": "Mean Episode Length",
    "rollout/exploration_rate": "Exploration Rate",
    "time/fps": "Frames Per Second"
}

ea = EventAccumulator(log_dir)
ea.Reload()

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, (tag, title) in enumerate(tags.items()):
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        axs[i].plot(steps, values)
        axs[i].set_title(title)
        axs[i].set_xlabel("Timesteps")
        axs[i].set_ylabel(title)
        axs[i].grid(True)

    except KeyError:
        axs[i].text(0.5, 0.5, f"Missing: {tag}", ha='center', va='center')
        axs[i].set_title(title)
        axs[i].axis('off')

plt.tight_layout()
plt.savefig(output_pdf)
print(f"✅ Saved {output_pdf}")
